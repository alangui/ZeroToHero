import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import matplotlib.pyplot as plt
import json
import math
import logging
from torch.utils.checkpoint import checkpoint
import sentencepiece as spm

batch_size = 64
block_size = 256
max_iters = 10000
eval_interval = 500
train_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 768
n_head = 12
n_layer = 12
dropout = 0.2
save_model_interval = 3000
cuda_mem_sum_interval = 2000

torch.manual_seed(1337)
# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('train.log', encoding='utf-8'),  # 写入文件
        logging.StreamHandler()  # 同时打印到终端
    ]
)

with open('../wiki_corpus_0.1b_clean.txt', 'r', encoding='utf-8') as f:
    text = f.read()

sp = spm.SentencePieceProcessor(model_file='zh_bpe.model')
## 此处注册一个用户自定义符号当"换行 token", 不然推理有问题，生成文本会变成一坨从头到尾不分段的文字
NL_ID = sp.piece_to_id('[BR]')

ids = []
for line in text.split('\n'):
    ids.extend(sp.encode(line))
    ids.append(NL_ID)
data = torch.tensor(ids, dtype=torch.long)
vocab_size = sp.vocab_size() 

n = int(0.9*len(data)) # 前90%的字符用于训练
train_data = data[:n]
val_data = data[n:]
logging.info(f"总样本数据 {len(data)}, 训练集 {len(train_data)} 验证集 {len(val_data)} 词元表大小:{vocab_size}")
# 数据加载
def get_batch(split):
    # 生成一小批数据，包含输入x和目标y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 生成一个形状为(batch_size,)的张量，包含0到len(data) - block_size之间的随机序列起始索引
    x = torch.stack([data[i:i+block_size] for i in ix])       # 将所有（ix包含batch_size个）序列按行堆叠在一起形成张量
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])   # 与x相同，但向后偏移一个token
    x, y = x.to(device), y.to(device)
    return x, y # x的维度是batch_size × block_size，y的维度也是batch_size × block_size

@torch.no_grad() # 为此函数禁用梯度计算
def estimate_loss(model):
    out = {}
    model.eval() # 将模型设置为评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 将模型恢复为训练模式
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        #wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        #wei = F.softmax(wei, dim=-1)
        #wei = self.dropout(wei)
        #out = wei @ self.value(x)
        #return out

        # 由于GPU 8G显存不够 对上面注释代码做如下优化，自动实现下三角掩码wei
        # PyTorch 内置的 F.scaled_dot_product_attention 底层用 Flash Attention 内核，从不物化完整注意力矩阵，还顺带用上 Tensor Core 提速。
        v = self.value(x)
        out = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0
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
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
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

    def forward(self, x):
        #x = x + self.sa_head(self.ln1(x))
        #x = x + self.ffwd(self.ln2(x))

        # 由于GPU 8G显存不够 对上面注释代码做如下优化
        # 反向传播时重新计算激活值而不是存着，能省掉 60%+ 的激活显存，代价是训练慢约 30%——但比起溢出后的 20 秒/步，这个交换血赚。
        # 前向时不再保存每一层的中间激活值，只存每层的输入；反向传播到某层时，临时重新跑一遍该层前向来重建激活值。用约 30% 的额外计算，换掉大部分激活值显存。对生成/评估无影响
        x = x + checkpoint(lambda t: self.sa_head(self.ln1(t)), x, use_reentrant=False)
        x = x + checkpoint(lambda t: self.ffwd(self.ln2(t)), x, use_reentrant=False)

        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 每个token直接从查找表中读取下一个token的logits
        self.token_embd = nn.Embedding(vocab_size, n_embd)     # 词汇嵌入，每个单独的token由一个大小为vocab_size × n_embd的向量表示
        self.position_embd = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embd(idx)
        pos_embd = self.position_embd(torch.arange(T, device=device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)                # 将logits重塑为(B*T, C)（B=batch_size, T=block_size, C=vocab_size）
            targets = targets.view(B*T)                 # 将targets重塑为(B*T)
            loss = F.cross_entropy(logits, targets)     # 计算整个批次中所有token的交叉熵损失

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx是当前上下文中形状为(B, T)的索引数组
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                          # 关注logits中的最后一个token，(B, T, C) -> (B, C)
            probs = F.softmax(logits, dim=-1)                  # 基于最后一个token计算下一个token的概率分布，结果为(B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # 采样下一个token（B, 1），概率最高的token最可能被采样到
            idx = torch.cat((idx, idx_next), dim=1)            # 将新token添加到序列中（B, T+1），用于下一次迭代
        return idx

    def generate_streaming(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                          # 关注logits中的最后一个token，(B, T, C) -> (B, C)
            probs = F.softmax(logits, dim=-1)                  # 基于最后一个token计算下一个token的概率分布，结果为(B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # 采样下一个token（B, 1），概率最高的token最可能被采样到
            idx = torch.cat((idx, idx_next), dim=1)            # 将新token添加到序列中（B, T+1），用于下一次迭代
            print(sp.decode(idx_next[0].tolist()).replace('[BR]', '\n'), end='', flush=True)
        print()

def get_lr(iter, max_iters, base_lr=3e-4, min_lr=3e-5, warmup=100):
    if iter < warmup:                      # 开头热身，缓慢上升
        return base_lr * (iter + 1) / warmup
    ratio = (iter - warmup) / (max_iters - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * ratio))

def train_loop():
    model = BigramLanguageModel()
    m = model.to(device)
    logging.info(f"{sum(p.numel() for p in m.parameters())} 个参数")
    # 创建PyTorch优化器
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    t_start = time.time()
    t_train_sum = 0.0
    t_train_start = 0.0
    losses_record = []

    scaler = torch.amp.GradScaler('cuda')
    # 训练循环
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            t_eval = time.time()
            losses = estimate_loss(model)
            losses_record.append({'train': float(losses['train']), 'val': float(losses['val'])})
            logging.info(f"第{iter}次训练后loss评估: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, loss评估耗时 {time.time() - t_eval:.1f} 秒") 

        if iter % train_interval == 0:
            t_train_start = time.time()

        xb, yb = get_batch('train')     # 获取批次

        with torch.autocast(device_type='cuda', dtype=torch.float16):   # 前向用 FP16
            _, loss = model(xb, yb)    # 前向传播

        opt.zero_grad(set_to_none=True)      # 重置梯度
        scaler.scale(loss).backward()        # 缩放后的反向传播
        scaler.step(opt)
        scaler.update()

        if iter % cuda_mem_sum_interval == 0:
            logging.info(torch.cuda.memory_summary())

        lr = get_lr(iter, max_iters)
        for g in opt.param_groups:
            g['lr'] = lr

        if (iter+1) % train_interval == 0:
            t_train_end = time.time() - t_train_start
            t_train_sum += t_train_end
            logging.info(f"第{iter+1}次train 完成, 本轮{train_interval}次训练耗时 {t_train_end:.1f} 秒, "
                  f"累计训练耗时 {t_train_sum/60:.1f} 分钟, 当前学习率 {lr:.2e}")

        if (iter+1) % save_model_interval == 0:    
            torch.save(model.state_dict(), f'model_final_zh_v6_{iter}.pt')    

    t_total = time.time() - t_start
    logging.info(f"全部训练完成，总耗时 {t_total/60:.1f} 分钟，" f"平均每步 {t_total/max_iters*1000:.0f} 毫秒")

    torch.save(model.state_dict(), f'model_final_zh_v6.pt')

    save_losses_json(losses_record)
    plot_losses(losses_record)

    return model

def train_one(model, opt):
    xb, yb = get_batch('train')     # 获取批次
    _, loss = model(xb, yb)         # 前向传播

    opt.zero_grad(set_to_none=True) # 重置梯度
    loss.backward()                 # 反向传播
    opt.step()                      # 更新参数


def load_mode_generate_txt_streaming():
    model = BigramLanguageModel()
    model.load_state_dict(torch.load('model_final_zh_v6.pt', map_location='cpu'))
    model.to(device)
    model.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    model.generate_streaming(context, max_new_tokens=1000)
    

def load_model_for_inference():
    model = BigramLanguageModel()
    model.load_state_dict(torch.load('model_final_zh_v6.pt', map_location='cpu'))
    model.to(device)
    model.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(sp.decode(model.generate(context, max_new_tokens=1000)[0].tolist()).replace('[BR]', '\n'), end='', flush=True)

def analy_model():
    ckpt = torch.load('model_final_zh_v6.pt', map_location='cpu', weights_only=True)

    for name, tensor in ckpt.items():
        print(f"{name:45s} {tuple(tensor.shape)}")

def plot_losses(losses_record, save_path='loss_curve_v6.png'):
    # x 轴：每次评估对应的训练步数（0, 500, 1000, ... 最后一步）
    steps = [i * eval_interval for i in range(len(losses_record))]
    train_losses = [l['train'] for l in losses_record]
    val_losses = [l['val'] for l in losses_record]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, marker='o', label='训练 loss')
    plt.plot(steps, val_losses, marker='s', label='验证 loss')
    plt.xlabel('训练步数')
    plt.ylabel('Loss')
    plt.title('Loss 变化趋势')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"趋势图已保存到 {save_path}")

def save_losses_json(losses_record):
    with open('losses_record_v6.json', 'w') as f:
        json.dump(losses_record, f)

def main():

    #train_loop()

    #load_model_for_inference()

    load_mode_generate_txt_streaming()

    #analy_model()

if __name__ == "__main__":
    main()