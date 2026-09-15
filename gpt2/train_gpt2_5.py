import os
import sys
import math
import time
import torch
import inspect
import torch.backends
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

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

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B # 批次大小
        self.T = T # 上下文长度

        # 初始化时从磁盘加载token并存储在内存中
        with open('../tiny-shakespeare.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text) # 将全文编码为token
        self.tokens = torch.tensor(tokens) # 用tensor包装
        self.tok_count = len(self.tokens)
        # 给我们这些极客看的一些统计信息
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # 每个下一个批次的token级起始位置
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # 截取大小为B * T + 1的一段token（之前解释过）
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) # 输入张量，大小为(B * T)
        y = buf[1:].view(B, T)  # 目标张量，大小为(B * T)，向右偏移1个位置
        # 在数据张量中前移位置
        self.current_position += B * T
        # 如果加载下一个批次会越界，则重置
        if self.current_position + (B * T + 1) >= len(self.tokens):
            # 将位置重置到数据开头
            self.current_position = 0
        return x, y

# -----------------------------------------------------------------------------

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
assert total_batch_size % (B * T) == 0, "Batch size must be divisible by (micro-batch size * sequence length)."
# 在grad_accum_steps个步数上累积梯度，而不是每一步都反向传播
grad_accum_steps = total_batch_size // (B * T) # 32个micro-batch组成1个macro-batch
print(f"Total desired batch size: {total_batch_size}")
print(f"-> Calculated gradient accumulation steps: {grad_accum_steps}")

# 用于micro-batch训练数据的DataLoaderLite
train_loader = DataLoaderLite(B=B, T=T)

# 矩阵乘法使用TF32张量浮点精度
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig()) # 随机权重初始化
model.to(device)

# 检查是否在Linux上运行：
if os.name == 'posix' and sys.platform != 'darwin':
    model = torch.compile(model) # 将模型编译为TorchScript -> 提升速度 + 节省内存
else:
    print("[!] Not running Linux - Skipping platform-unsupported torch.compile()")


max_lr = 6e-4 # 根据GPT-3论文
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

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
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
# 你可以把Adam看作是RMSprop和带动量的随机梯度下降（SGD）的结合，即一个更精细的SGD版本。
# AdamW是Adam的一个版本，它对权重衰减有更好的实现。在大多数情况下你可以直接用AdamW代替Adam。
# https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
# 就这个例子而言，我们直接用它，某种程度上把它当作一个黑盒。

# 优化循环
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad() # 每个macro-batch重置梯度
    loss_accum = 0.0 # 跨micro-batch的损失累加器
    # micro-batch的内层循环
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch() # (B, T)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # 按完整批次大小与micro-batch大小的比例归一化micro-batch损失
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        # 现在梯度会累积，因为我们在内层循环中没有调用zero_grad()
        loss.backward()
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
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps # 每步处理的token数
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")      

#import sys; sys.exit(0)

# 生成！现在x是(B, T)，其中B = 5，T = 8
model.eval()
num_return_sequences = 5
max_length = 30
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

while x.size(1) < max_length:
    # 前向传播模型以获取logits
    with torch.no_grad():
        logits, _ = model(x) # (B, T, vocab_size)
        # 取最后一个位置的logits
        logits = logits[:, -1, :] # (B, vocab_size)
        # 获取概率
        probs = F.softmax(logits, dim=-1)
        # 进行top-k采样，k=50（huggingface pipeline的默认值）
        # 这里topk_probs变为(5, 50)，topk_indices为(5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # 从top-k概率中选择一个token
        # 注意：multinomial不要求输入概率之和为1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # 收集对应的索引
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # 追加到序列中
        x = torch.cat((x, xcol), dim=1) # (B, T+1)

# 打印生成的文本
for step in range(num_return_sequences):
    tokens = x[step, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
