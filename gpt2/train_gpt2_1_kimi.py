# train_gpt2.py —— 跟随 Karpathy "Let's reproduce GPT-2 (124M)" 的复现脚本（中文注释版）
# 与根目录 gpt.py 的字符级小模型不同，这里按 GPT-2 论文结构实现：
# BPE 分词（tiktoken gpt2）+ 绝对位置嵌入 + pre-norm 残差块 + 权重共享（tying）

import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# 超参数（文件顶部模块级常量，与项目其他脚本保持同一风格）
batch_size = 16          # MPS 下 124M 模型能跑起来的保守值；有 CUDA 可开到 32+
block_size = 1024        # GPT-2 的上下文长度
max_iters = 2000
eval_interval = 200
eval_iters = 20
learning_rate = 6e-4     # GPT-2 small 的峰值学习率
min_lr = 6e-5            # 余弦退火的下限（峰值的 1/10）
warmup_iters = 100
weight_decay = 0.1
grad_clip = 1.0
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

torch.manual_seed(1337)


class GPTConfig:
    # GPT-2 (124M) 的标准配置
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # q/k/v 合并成一个投影，一次矩阵乘出 3 份，比三个 Linear 略快
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # GPT-2 对残差路径上的投影层用特殊缩放初始化（见 _init_weights）
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, T, C) -> (B, nh, T, hs)，让每个头独立做注意力
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Flash Attention：is_causal=True 自带因果掩码，不物化 T×T 注意力矩阵
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')  # GPT-2 用 tanh 近似的 GELU
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    # pre-norm 结构：先 LayerNorm 再进注意力/MLP，残差旁路保持干净

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

    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # GPT-2 的 tying：token 嵌入与输出投影共享同一份权重
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # GPT-2 论文：残差路径上的投影层按 1/sqrt(2*n_layer) 缩小初始化，
            # 控制残差流随深度增长的方差
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"序列长度 {T} 超过 block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)   # (B, T, C)
        pos_emb = self.transformer.wpe(pos)   # (T, C)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)              # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # 只给 2 维以上参数（矩阵）加 weight decay，bias/LayerNorm/Embedding 不加
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        print(f"weight decay 参数张量数: {len(decay_params)}，不 decay 参数张量数: {len(nodecay_params)}")
        # fused 目前只在 CUDA 上可用，其他设备自动退回普通 AdamW
        use_fused = device == 'cuda' and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]   # 超长就截断到 block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class DataLoaderLite:
    # 用 tiktoken 的 GPT-2 BPE 把文本整体编码成 token 流，按 batch 顺序切片
    def __init__(self, B, T, path='../tiny-shakespeare.txt'):
        import tiktoken
        self.B, self.T = B, T
        enc = tiktoken.get_encoding('gpt2')
        self.enc = enc
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        n = int(0.9 * len(self.tokens))
        self.train_tokens = self.tokens[:n]
        self.val_tokens = self.tokens[n:]
        self.current_position = 0
        print(f"共 {len(self.tokens)} 个 token，train {len(self.train_tokens)} / val {len(self.val_tokens)}")

    def next_batch(self, split='train'):
        data = self.train_tokens if split == 'train' else self.val_tokens
        buf = data[self.current_position:self.current_position + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_position += self.B * self.T
        if self.current_position + self.B * self.T + 1 > len(data):
            self.current_position = 0   # 一轮读完回到开头
        return x.to(device), y.to(device)


def get_lr(it):
    # 线性 warmup + 余弦退火到 min_lr
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it >= max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, loader):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = loader.next_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_loop():
    model = GPT(GPTConfig()).to(device)
    print(f"设备: {device}，参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    loader = DataLoaderLite(B=batch_size, T=block_size)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, device)

    t_start = time.time()
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, loader)
            print(f"第 {it} 步评估: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        x, y = loader.next_batch('train')
        # MPS 支持 bf16 autocast；显存/算力紧张的设备上能省不少内存
        if device in ('cuda', 'mps'):
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lr = get_lr(it)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()

        if (it + 1) % 100 == 0:
            print(f"第 {it+1} 步 train loss {loss.item():.4f}, lr {lr:.2e}, 累计耗时 {(time.time()-t_start)/60:.1f} 分钟")

    torch.save(model.state_dict(), 'model_gpt2_final.pt')
    print(f"训练完成，总耗时 {(time.time()-t_start)/60:.1f} 分钟，权重已存到 model_gpt2_final.pt")
    return model


def load_model_generate():
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    model = GPT(GPTConfig())
    model.load_state_dict(torch.load('model_gpt2_final.pt', map_location='cpu', weights_only=True))
    model.to(device)

    prompt = "To be or not to be,"
    idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=200)
    print(enc.decode(out[0].tolist()))


def analy_model():
    # sanity check：打印权重形状
    ckpt = torch.load('model_gpt2_final.pt', map_location='cpu', weights_only=True)
    for name, tensor in ckpt.items():
        print(f"{name:45s} {tuple(tensor.shape)}")


def main():
    train_loop()

    #load_model_generate()

    #analy_model()

if __name__ == "__main__":
    main()
