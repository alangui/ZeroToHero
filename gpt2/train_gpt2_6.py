import os
import sys
import math
import time
import torch
import inspect
import tiktoken
import numpy as np
import torch.backends
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import torch.nn.parallel.DistributedDataParallel as DDP
from torch.distributed import dist, init_process_group, destroy_process_group
from hellaswag import iterate_examples, render_example


# -----------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024  # 最大序列长度
    vocab_size: int = 50304 # token数量：50,000个BPE合并 + 256个字节token + 1个<|endoftext|> token
    n_layer: int = 12       # 层数
    n_head: int = 12        # 头数
    n_embd: int = 768       # 嵌入维度

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 所有头的key、query、value投影，但合并为一个批次
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # 很喜欢这种写法哈哈
        # 正则化
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 实际上不是'bias'（偏置），更像是一个掩码，但沿用了OpenAI/HF的命名
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # 批次大小、序列长度、嵌入维度（n_embd）
        # 批量计算所有头的query、key、values，并将head维度提前作为batch维度
        # nh是"头数"，hs是"头大小"，C（通道数）= nh * hs
        # 例如在GPT-2（124M）中，n_head=12，hs=64，所以nh*hs=C=768是Transformer中的通道数
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # 注意力（会为所有query和key生成巨大的(T,T)矩阵）
        
        # 详细展开的注意力实现
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v #(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # 我们改用Flash Attention实现（在Win32上可能会抛出警告）
        # https://www.reddit.com/r/comfyui/comments/1cerq2e/is_uh_comfyanon_aware_that_pytorch_flash/
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # 将所有头的输出按顺序拼接在一起，形状为(B, T, C)
        # 输出投影
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 权重共享方案
        self.transformer.wte.weight = self.lm_head.weight
        # 本质上是遍历所有子模块并对其应用该函数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 线性层权重以均值0、标准差0.02的高斯分布初始化
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 标准差按1 / sqrt(N)缩放
                # 乘以2 * n_layer，因为每个block中都有注意力层*和*MLP（block数由n_layer计）
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                # 偏置显式初始化为0
                torch.nn.init.zeros_(module.bias)
        # 嵌入层权重以均值0、标准差0.02的高斯分布初始化
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx和targets的形状均为(B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # 前向传播token嵌入和位置嵌入
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # 形状 (T)
        pos_emb = self.transformer.wpe(pos) # 位置嵌入，形状为(T, n_embd)
        tok_emb = self.transformer.wte(idx) # token嵌入，形状为(B, T, n_embd)
        x = tok_emb + pos_emb
        # 前向传播transformer的各个block
        for block in self.transformer.h:
            x = block(x)
        # 前向传播最终的layernorm和分类器
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """从huggingface加载预训练的GPT-2模型权重"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer、n_head和n_embd由model_type决定
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M参数
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M参数
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M参数
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M参数
        }[model_type]
        config_args['vocab_size'] = 50257 # GPT模型检查点始终为50257
        config_args['block_size'] = 1024 # GPT模型检查点始终为1024
        # 创建一个从零初始化的minGPT模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # 丢弃这个掩码/缓冲区，它不是参数

        # 初始化一个HuggingFace/Transformers模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 复制参数，同时确保所有参数在名称和形状上对齐匹配
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # 忽略这些，只是缓冲区
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # 同样，只是掩码（缓冲区）
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 基本上OpenAI的检查点使用"Conv1D"模块，但我们只想使用普通的Linear
        # 这意味着导入这些权重时必须对它们进行转置
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 对需要转置的Conv1D权重进行特殊处理
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 普通方式复制其他参数
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # 从所有候选参数（需要梯度的）开始
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # 创建优化器分组。任何2D及以上的参数都会进行权重衰减，否则不衰减。
        # 即所有矩阵乘法中的权重张量 + 嵌入层进行衰减，但所有偏置和layernorm不衰减。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")
        # 创建AdamW优化器，如果可用则使用fused版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = torch.cuda.is_available() and fused_available and device.startswith('cuda')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # 见勘误表
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B # 批次大小
        self.T = T # 上下文长度
        self.process_rank = process_rank # 当前进程的rank（序号）
        self.num_processes = num_processes # 进程总数
        assert split in {'train', 'val'}, "split must be one of 'train' or 'val'"

        # 获取分片文件名
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split '{split}' in '{data_root}'"
        if master_process:
            print(f"found {len(shards)} shards for split '{split}'")
        self.reset()

    def reset(self):
        # 状态，从第0个分片开始初始化
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.tok_count = len(self.tokens)
        self.current_position = self.B * self.T * self.process_rank    

    def next_batch(self):
        B, T = self.B, self.T
        # 截取大小为B * T + 1的一段token（之前解释过）
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) # 输入张量，大小为(B * T)
        y = buf[1:].view(B, T)  # 目标张量，大小为(B * T)，向右偏移1个位置
        # 在数据张量中前移位置
        # 新增：跳转到下一个批次，考虑进程数量
        self.current_position += B * T * self.num_processes
        # 如果加载下一个批次会越界，则重置
        # 新增：检查考虑并行进程后的跳转是否会越界
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            # 相对于进程rank将位置重置到数据开头
            self.current_position = self.B * self.T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------

# HellaSwag评估的辅助函数
# 接收tokens、mask和logits，返回损失最低的补全项的索引

def get_most_likely_row(tokens, mask, logits):
    # 评估所有位置上的自回归损失
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # 现在计算每一行中仅补全区域（mask == 1）的平均损失
    shift_mask = (mask[..., 1:]).contiguous() # 我们必须偏移mask，使其从最后一个prompt token开始
    masked_shift_losses = shift_losses * shift_mask
    # 求和并除以mask中1的个数
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # 现在我们得到了B个补全项各自的损失
    # 损失最低的那个应该是最可能的
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------

# 设置DDP（分布式数据并行）环境
ddp = int(os.environ.get('RANK', -1)) != -1 # 检查我们是否处于DDP环境/运行中
if ddp:
    # 目前使用DDP要求CUDA，我们根据rank适当设置设备
    assert torch.cuda.is_available(), "DDP requires CUDA for now - please run on a GPU node."
    init_process_group(backend='nccl')  # 初始化分布式后端
    ddp_rank = int(os.environ['RANK'])  # 当前进程的rank
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 当前进程的本地rank/GPU索引
    # （ddp_local_rank指节点内，ddp_rank是全局的，即跨所有节点；节点：就是一台配有多个GPU的机器（顺便说一下））
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # DDP环境中的进程数
    device = f"cuda:{ddp_local_rank}"  # 根据本地rank映射设备（指示在节点上使用哪个GPU）
    torch.cuda.set_device(device)      # 将设备设置为本地rank对应的GPU
    master_process = ddp_rank == 0     # 检查当前进程是否是主进程（主进程负责日志记录、检查点保存等）
else:
    # 普通的非DDP运行
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # 找到最适合训练的设备
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_initialized():
        device = "mps" # Apple Silicon
    print(f"Using device: {device}")

# 可复现性
torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2**19，约0.5M个token，遵循GPT-3论文
# 每个micro-batch处理16384个token
# -> 需要32个（grad_accum_steps）micro-batch累积成一个完整批次
B = 4 # micro-batch大小
T = 1024 # 序列长度
assert total_batch_size % (B * T * ddp_world_size) == 0, "Batch size must be divisible by (micro-batch size * sequence length * ddp_world_size)."
# 在grad_accum_steps个步数上累积梯度，而不是每一步都反向传播
# 每个进程处理B * T个token，共有ddp_world_size个进程
# 例如在8个GPU上，反向传播前我们每步将在16 * 1024 * [8] = 131072个token上学习
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # 32个micro-batch组成1个macro-batch

if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"-> Calculated gradient accumulation steps: {grad_accum_steps}")

# 分别用于micro-batch训练数据和验证数据的DataLoaderLite
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# 矩阵乘法使用TF32张量浮点精度
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig()) # 随机权重初始化
model.to(device)

# 检查是否在Linux上运行：
if os.name == 'posix' and sys.platform != 'darwin':
    model = torch.compile(model) # 将模型编译为TorchScript -> 提升速度 + 节省内存
else:
    print("[!] Not running Linux - Skipping platform-unsupported torch.compile()")

if ddp:
    # 每个GPU的反向传播完成后，DDP会在所有GPU间平均梯度
    model = DDP(model, device_ids=[ddp_local_rank])

# DDP需要原始模型的引用以进行正确处理
raw_model = model.module if ddp else model

max_lr = 6e-4 # 根据GPT-3论文
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 * 4

def get_lr(it):
    if it < warmup_steps:
        # 1) warmup_iters步的线性预热区间
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        # 2) 如果it > lr_decay_iters，直接返回min_lr
        return min_lr
    # 3) 在预热和max_steps之间，余弦衰减到min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # 系数从1开始并降至0
    return min_lr + coeff * (max_lr - min_lr)


# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
# 你可以把Adam看作是RMSprop和带动量的随机梯度下降（SGD）的结合，即一个更精细的SGD版本。
# AdamW是Adam的一个版本，它对权重衰减有更好的实现。在大多数情况下你可以直接用AdamW代替Adam。
# https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
# 就这个例子而言，我们直接用它，某种程度上把它当作一个黑盒。

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
# 将包含训练过程中的训练损失、验证损失和hellaswag准确率
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    # 清空日志文件
    pass

# 加载分词器
enc = tiktoken.get_encoding('gpt2')

# 优化循环
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1) # 检查是否是最后一步
    # 每隔一段时间，在验证集上检查
    if step % model.config.eval_iter == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.6f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.6f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # 可选择地写出检查点
                checkpoint_path = os.path.join(log_dir, f"checkpoint_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    # 添加这些以便能够轻松地从检查点恢复训练
                    'optimizer': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state(),
                }
                torch.save(checkpoint, checkpoint_path)

    # 每隔一段时间，评估hellaswag数据集
    if (step % model.config.eval_iter == 0 or last_step) and (not use_compile):
        model.eval()
        num_correct_norm = 0
        num_total = 0

        for i, example in enumerate(iterate_examples("val")):
            # 只处理满足i % ddp_world_size == ddp_rank的样本
            if i % ddp_world_size != ddp_rank:
                continue
            # 将样本渲染为tokens和标签
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # 获取logits
            with torch.no_grad():
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    logits, _ = model(tokens, mask)
                pred_norm = get_most_likely_row(tokens, mask, logits) # 获取批次中每个序列最可能的补全项
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag acc_norm: {num_correct_norm}/{num_total}={acc_norm:.6f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hellaswag {acc_norm:.6f}\n")

    # 每隔一段时间，生成一些文本
    if ((step > 0 and step % model.config.eval_iter == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank) # 为可复现性设置随机数生成器的种子
        while xgen.size(1) < max_length:
            # 前向传播模型以获取logits
            with torch.no_grad():
                logits, loss = model(xgen) # (B, T, vocab_size)
                # 取最后一个位置的logits
                logits = logits[:, -1, :] # (B, vocab_size)
                # 获取概率
                probs = F.softmax(logits, dim=-1)
                # 进行top-k采样，k=50（huggingface pipeline的默认值）
                # 这里topk_probs变为(5, 50)，topk_indices为(5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # 从top-k概率中选择一个token
                # 注意：multinomial不要求输入概率之和为1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # 收集对应的索引
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # 追加到序列中
                xgen = torch.cat((xgen, xcol), dim=1) # (B, T+1)
        # 打印生成的文本
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    model.train()
    optimizer.zero_grad() # 每个macro-batch重置梯度
    loss_accum = 0.0 # 跨micro-batch的损失累加器
    # micro-batch的内层循环
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch() # (B, T)
        x, y = x.to(device), y.to(device)
        if ddp:
            # DDP会负责在所有GPU间平均损失
            # DDP只会在每个累积周期的最后一个micro-batch之后执行平均
            # 移到这里是因为前向和反向传播都需要它
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # 按完整批次大小与micro-batch大小的比例归一化micro-batch损失
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        # 现在梯度会累积，因为我们在内层循环中没有调用zero_grad()
        loss.backward()
    if ddp:
        # 在所有GPU间平均macro-batch损失，
        # 完成平均后，同步该平均损失，使其在所有GPU上具有相同的值
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 将全局梯度范数裁剪到1.0
    # 确定并设置本次迭代的学习率
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        # 这里有点像是侧向注入我们实际想要应用到参数上的学习率
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # 等待GPU完成上面调度的工作负载
    t1 = time.time()
    dt = (t1 - t0) # 毫秒级时间差
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size # 所有GPU每步处理的token总数
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            # 将训练损失写入文件
            f.write(f"{step} train {loss_accum.item():.6f}\n")     

if ddp:
    destroy_process_group() # DDP清理

