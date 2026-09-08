# eval_on_benchmark.py —— 在永久基准集上给任意 checkpoint 评分
import torch
import torch.nn as nn
from torch.nn import functional as F

MODEL_PATH = '../中文版-v6-gpu/model_final_zh_v6.pt'   # ← 只改这一行
BENCH_PATH = 'benchmark_val.pt'    # 基准集在 ZeroToHero 根目录

# ===== 与训练脚本完全一致的结构配置（v8~v10 相同）=====
batch_size = 32
block_size = 256
n_embd = 768
n_head = 12
n_layer = 12
dropout = 0.2
vocab_size = 16000                    # 与 zh_bpe.model 一致
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        out = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
            is_causal=True, dropout_p=0.0   # 评估永远关 dropout
        )
        return out.squeeze(1)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_head = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):   # 评估不需要梯度检查点，直接前向
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embd = nn.Embedding(vocab_size, n_embd)
        self.position_embd = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embd(idx)
        pos_embd = self.position_embd(torch.arange(T, device=device))
        x = self.blocks(tok_embd + pos_embd)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

# ===== 评估 =====
bench = torch.load(BENCH_PATH)
print(f'基准集 token 数: {len(bench):,}')

model = BigramLanguageModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.to(device).eval()

losses = torch.zeros(50)
with torch.no_grad():
    for k in range(50):
        ix = torch.randint(len(bench) - block_size, (batch_size,))
        x = torch.stack([bench[i:i+block_size] for i in ix]).to(device)
        y = torch.stack([bench[i+1:i+block_size+1] for i in ix]).to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, l = model(x, y)
        losses[k] = l.item()
print(f'{MODEL_PATH}')
print(f'基准集 val loss: {losses.mean():.4f} ± {losses.std():.4f}')