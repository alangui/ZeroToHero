from dataclasses import dataclass
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

@dataclass
class GPTConfig:
    # GPT-2 (124M) 的标准配置
    block_size: int = 1024
    vocab_size: int = 50257
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
        # 注意力（会实际生成那个针对所有 query 与 key 的大 (T, T) 矩阵）
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
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

    def forward(self, idx):
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
        return logits

    
num_return_sequences = 5
max_length = 16

model = GPT(GPTConfig())
model.eval()
model.to(device)

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,") #(8,) 一维张量
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(5,8) 二维张量
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # 只取最后一个位置的 logits（能用，但有浪费）
        logits = logits[:, -1, :] # (B, vocab_size)
        # 得到概率
        probs = F.softmax(logits, dim=-1)
        # 做 top-k = 50 的采样（huggingface pipeline 的默认值），top-k 采样的思路：只在概率最高的前 50 个候选里随机选，尾巴直接砍掉。
        # 这里 topk_probs 变成 (5, 50)，topk_indices 是 (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # 从 top-k 概率中选出一个 token
        ix = torch.multinomial(topk_probs, 1) # (8, 1)
        # 取出对应的索引
        xcol = torch.gather(topk_indices, -1, ix) # (8, 1)
        # 追加到序列上
        x = torch.cat((x, xcol), dim=1) # (8, T+1)

# 打印生成的文本
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)