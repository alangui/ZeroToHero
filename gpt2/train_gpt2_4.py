from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import time
from contextlib import nullcontext
import os
import sys
import math
import inspect

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


@dataclass
class GPTConfig:
    # GPT-2 (124M) 的标准配置
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 所有 head 的 Key、Query、Value 投影，但合并成一批
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # 正则化相关
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 其实不算 'bias'，更像是一个掩码，不过这里沿用 OpenAI/HF 的命名
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))        

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        """
        # 注意力（会实际生成那个针对所有 query 与 key 的大 (T, T) 矩阵）
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        """
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
    

class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GPT-2 用 tanh 近似的 GELU
        self.gelu = nn.GELU(approximate='tanh')  
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x =self.c_proj(x)
        return x  


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super(Block, self).__init__()
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
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Linear weights distributed with mean 0 and std 0.02
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 1 / sqrt(N) scaling fo the std dev
                # 2 * n_layer because we have attention *and* mlp per block (blocks counted by n_layer)
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                # Biases explicitly initialized to 0
                torch.nn.init.zeros_(module.bias)
        # Embedding weights distributed with mean 0 and std 0.02
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)        

    def forward(self, idx, targets=None):
        # idx 的形状为 (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        # 前向计算 token embedding 与位置 embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # 形状为 (T)
        pos_emb = self.transformer.wpe(pos) # 位置 embedding，形状 (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embedding，形状 (B, T, n_embd)
        x = tok_emb + pos_emb # 相加合并
        # 前向穿过各 transformer block
        for block in self.transformer.h:
            x = block(x)
        # 前向穿过最终层归一化与分类头
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, but all biases and layernorms don't.
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
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = torch.cuda.is_available() and fused_available and device.startswith('cuda')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


class DataLoaderLite:

    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('../tiny-shakespeare.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.tok_count = len(self.tokens)
        print(f"loaded {self.tok_count} tokens")
        print(f"1 epoch = {self.tok_count // (B * T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position +  (B * T + 1) >= len(self.tokens):
            self.current_position = 0
        return x, y

model = GPT(GPTConfig())
model.to(device)

if os.name == 'posix' and sys.platform != 'darwin':
    model = torch.compile(model) # compile model to TorchScript -> speed + memory savings
else:
    print("[!] Not running Linux - Skipping platform-unsupported torch.compile()")

max_lr = 6e-4 # According to GPT-3 paper
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    if it < warmup_steps:
        # 1) Linear warmup region for warmup_iters steps
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        # 2) if it > lr_decay_iters, flat out return the min_lr
        return min_lr
    # 3) In between warmup and max_steps, cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to zero
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)

total_batch_size = 4096
B = 4
T = 128
assert total_batch_size % (B * T) == 0, "Batch size must be divisible by (micro-batch size * sequence length)."
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch size: {total_batch_size}")
print(f"-> Calculated gradient accumulation steps: {grad_accum_steps}")


train_loader = DataLoaderLite(B=B, T=T)
ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
for step in range(50):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for miro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with ctx:
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        #Kind of side-injecting the actual learning rate we want to apply to the parameters here
        param_group['lr'] = lr
    optimizer.step()
    if device in ('cuda', 'mps'): getattr(torch, device).synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
