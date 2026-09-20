# -*- coding: utf-8 -*-
# sft_zh_v11.py —— 基于 v10 基座的指令微调（SFT），产出可对话的 GPT 助手
#
# 相对 v10 的唯一变量：+SFT（指令数据 + 词表扩维 + loss mask）。
# 模型骨架与 v8~v10 完全一致（16 层 × 1024 维 × 16 头，234.5M 参数，无 tying），
# 唯一结构差异是词表从 16000 扩到 16003（多出来的 3 行是 USER/ASSISTANT/EOT）。
#
# 运行方式（项目惯例）：cd 进本目录再跑，main() 里注释切换入口函数。
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import math
import logging
import random
import sentencepiece as spm
from torch.utils.checkpoint import checkpoint

# ========== 超参数（模块级常量，项目惯例）==========
batch_size = 16            # SFT 样本短且变长，batch 不用预训练那么大
block_size = 256           # 与 v10 一致，先跑通再消融扩 512
max_iters = 2000           # SFT 步数远少于预训练（几万条指令 × 2 epoch 撑死几千步）
eval_interval = 100
learning_rate = 2e-5       # 基座已训好，LR 比预训练（3e-4）低一个数量级，防灾难性遗忘
min_learning_rate = 1e-6
warmup_steps = 30
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 1024
n_head = 16
n_layer = 16
dropout = 0.1              # 预训练 0.2，SFT 数据小但目标是记住格式，降到 0.1
val_ratio = 0.05           # SFT 数据量少，只切 5% 做验证
seed = 1337

PRETRAIN_PATH = '../中文版-v10-gpu/model_best_zh_v10.pt'   # v10 最优基座（推理务必用 best）
SFT_DATA_PATH = 'sft_data.pt'                              # prepare_sft_data.py 的产物

# ========== 词表扩维（与 prepare_sft_data.py 严格一致）==========
SP_VOCAB_SIZE = 16000      # zh_bpe.model 原生词表
USER_ID = 16000            # <|user|>
ASSISTANT_ID = 16001       # <|assistant|>
EOT_ID = 16002             # <|eot|> 对话结束——模型学会它才会"收口"
vocab_size = SP_VOCAB_SIZE + 3
PAD_ID = 0                 # 批次内补齐用，labels 是 -100 不会算 loss，值无所谓

torch.manual_seed(seed)
random.seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('train.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

sp = spm.SentencePieceProcessor(model_file='zh_bpe.model')
NL_ID = sp.piece_to_id('[BR]')
assert sp.vocab_size() == SP_VOCAB_SIZE, f'词表大小异常: {sp.vocab_size()}'


def encode_text(t):
    """与预训练一致的编码：逐行 encode + 行间 [BR]（同 prepare_sft_data.py）"""
    return [i for line in sp.encode(t.split('\n')) for i in line + [NL_ID]]


def decode_ids(ids):
    """过滤掉扩展区特殊符号再解码（sp.decode 对越界 id 会抛 IndexError，实测确认）"""
    return sp.decode([i for i in ids if i < SP_VOCAB_SIZE]).replace('[BR]', '\n')


# ========== 模型骨架（与 v10 相同，只有词表维度不同）==========
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

    def forward(self, x):
        # 梯度检查点重新打开：2026-09-19 实测 batch 16 不开检查点 OOM（8.58GB 顶满，
        # v10 在同款卡上靠检查点压到 6464 MiB）。慢 30% 换显存安全，2000 步短跑可接受。
        x = x + checkpoint(lambda t: self.sa_head(self.ln1(t)), x, use_reentrant=False)
        x = x + checkpoint(lambda t: self.ffwd(self.ln2(t)), x, use_reentrant=False)
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
            loss = None
        else:
            B, T, C = logits.shape
            # ignore_index=-100：用户部分和 padding 的 labels 都是 -100，天然被跳过
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T),
                                   ignore_index=-100)
        return logits, loss


def load_pretrained_model():
    """加载 v10 基座 + 词表扩维手术。

    为什么不用 load_state_dict(strict=False)：它只放过缺 key，不放过后尾shape不匹配，
    而 v11 的 token_embd/lm_head 比 v10 多 3 行，直接 load 会报 size 错误。
    所以手动拷贝前 16000 行，其余 key 严格加载（严格是为了让结构漂移立刻报错）。
    新增 3 行用小初始化（N(0, 0.02)，与 GPT-2 标准初始化同量级），从头学起。
    """
    model = BigramLanguageModel()
    ckpt = torch.load(PRETRAIN_PATH, map_location='cpu', weights_only=True)

    embd_keys = {'token_embd.weight', 'lm_head.weight', 'lm_head.bias'}
    with torch.no_grad():
        model.token_embd.weight[:SP_VOCAB_SIZE].copy_(ckpt['token_embd.weight'])
        model.lm_head.weight[:SP_VOCAB_SIZE].copy_(ckpt['lm_head.weight'])
        model.lm_head.bias[:SP_VOCAB_SIZE].copy_(ckpt['lm_head.bias'])
        # 新符号行初始化：embedding 和 lm_head 是两份独立权重（v7 起无 tying），都要扩
        model.token_embd.weight[SP_VOCAB_SIZE:].normal_(0, 0.02)
        model.lm_head.weight[SP_VOCAB_SIZE:].normal_(0, 0.02)
        model.lm_head.bias[SP_VOCAB_SIZE:].zero_()

    rest = {k: v for k, v in ckpt.items() if k not in embd_keys}
    missing, unexpected = model.load_state_dict(rest, strict=False)
    # rest 里没有三个扩维 key 是预期行为；其余 key 必须一个不缺、一个不 Surprise
    assert not unexpected and set(missing) == embd_keys, \
        f'基座结构与 v11 模型不一致: missing={missing}, unexpected={unexpected}'

    logging.info(f'基座 {PRETRAIN_PATH} 加载完成，词表扩维 {SP_VOCAB_SIZE} → {vocab_size}，'
                 f'新增符号 id: USER={USER_ID} ASSISTANT={ASSISTANT_ID} EOT={EOT_ID}')
    return model


# ========== 数据加载（变长样本，padding + loss mask）==========
samples = None   # train_loop 里加载，避免推理入口白等

def load_samples(path=SFT_DATA_PATH):
    data = torch.load(path, weights_only=False)
    random.shuffle(data)
    n_val = max(1, int(len(data) * val_ratio))
    return data[n_val:], data[:n_val]


def get_batch(split_data):
    """从样本列表随机取一个 batch，padding 到批内最长长度。
    y 就是 labels（含 -100），shift 发生在 prepare 阶段已经处理好（ids/labels 等长对齐），
    这里直接 x=ids[:-1], y=labels[1:]，与预训练的 next-token 对齐方式一致。
    """
    picks = random.sample(split_data, batch_size)
    maxlen = min(max(len(s['ids']) for s in picks), block_size)
    x = torch.full((batch_size, maxlen - 1), PAD_ID, dtype=torch.long)
    y = torch.full((batch_size, maxlen - 1), -100, dtype=torch.long)
    for n, s in enumerate(picks):
        ids, labels = s['ids'][:maxlen], s['labels'][:maxlen]
        x[n, :len(ids) - 1] = torch.tensor(ids[:-1])
        y[n, :len(labels) - 1] = torch.tensor(labels[1:])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split, d in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(d)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ========== 对话推理 ==========
def build_chat_context(history, new_q):
    """把多轮历史 + 新问题拼成 id 序列。

    history: [(q, a), ...]，new_q: str。与训练样本格式逐 token 对齐，
    这是 SFT 后能否正常对话的生命线——训练和推理的格式差一个 token 都会显著掉质量。
    """
    ids = []
    for q, a in history:
        ids += [USER_ID] + encode_text(q) + [ASSISTANT_ID] + encode_text(a) + [EOT_ID]
    ids += [USER_ID] + encode_text(new_q) + [ASSISTANT_ID]
    return ids[-block_size:]


@torch.no_grad()
def generate_chat(model, context_ids, max_new_tokens=256, temperature=0.4, top_k=30,
                  repetition_penalty=1.2):
    """temperature/top-k 采样 + 重复惩罚；遇到 EOT 立即收口。

    2026-09-19 盲测教训：234M 弱模型用 temperature 0.7/top_k 50 采样噪声盖过正确
    token，输出复读机+乱码；降到 0.4/30 后稳定性显著改善。弱模型宁低勿高。
    2026-09-20 v11.2 盲测教训：弱模型开放式生成会掉进"秋风吹拂×N"式退化循环
    （循环路径自我强化，EOT 永远采不出来）。repetition_penalty=1.2 把已出现
    token 的 logits 压低（HF generate 同款机制），人为打破循环，不必重训。
    """
    model.eval()
    idx = torch.tensor([context_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if repetition_penalty != 1.0:
            # 出现过的 token：正 logits 除以惩罚、负 logits 乘惩罚，统一压低其概率
            seen = set(idx[0].tolist())
            for tid in seen:
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty
        logits = logits / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        if idx_next.item() == EOT_ID:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    # 只解码新生成的部分——上下文是输入，不属于回答，解码它会导致每轮重复打印历史
    return decode_ids(idx[0][len(context_ids):].tolist())


def chat(model_path='model_best_zh_v11.pt'):
    """命令行多轮对话。手动输入 exit 退出；空输入用内置问题快速冒烟。"""
    model = BigramLanguageModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.to(device).eval()

    history = []
    print('中文 GPT 助手 v11（输入 exit 退出，直接回车用内置问题）')
    while True:
        try:
            q = input('\n[你] ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q == 'exit':
            break
        if not q:
            q = '你好，介绍一下你自己'
            print(f'（内置问题：{q}）')
        context = build_chat_context(history, q)
        a = generate_chat(model, context)
        print(f'[助手] {a}')
        history.append((q, a))
        if len(history) > 20:      # 历史太长会顶掉 block_size，留最近 20 轮足够
            history = history[-20:]


# ========== 训练循环 ==========
def get_lr(iter):
    if iter < warmup_steps:
        return learning_rate * (iter + 1) / warmup_steps
    ratio = (iter - warmup_steps) / (max_iters - warmup_steps)
    return min_learning_rate + 0.5 * (learning_rate - min_learning_rate) * (1 + math.cos(math.pi * ratio))


def train_loop():
    train_data, val_data = load_samples()
    logging.info(f'SFT 样本: 训练 {len(train_data)} 条，验证 {len(val_data)} 条')

    model = load_pretrained_model().to(device)
    logging.info(f'{sum(p.numel() for p in model.parameters()):,} 个参数')
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    losses_record = []
    best_val = float('inf')
    best_iter = -1
    t_start = time.time()

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            losses_record.append({'train': float(losses['train']), 'val': float(losses['val'])})
            logging.info(f'第{iter}步评估: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

            # best 保存：SFT 步数少，不设 v10 那种"最后 1/4 进程"门槛，任何一步新低都存
            cur_val = float(losses['val'])
            if cur_val < best_val:
                best_val = cur_val
                best_iter = iter
                torch.save(model.state_dict(), 'model_best_zh_v11.pt')
                logging.info(f'val loss 创新低，已保存最优模型: best val {best_val:.4f} @ 第{best_iter}步')

        lr = get_lr(iter)
        for g in opt.param_groups:
            g['lr'] = lr

        xb, yb = get_batch(train_data)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if iter % 50 == 0:
            logging.info(f'第{iter}步, loss {loss.item():.4f}, lr {lr:.2e}, '
                         f'已耗时 {(time.time() - t_start) / 60:.1f} 分钟')

    torch.save(model.state_dict(), 'model_final_zh_v11.pt')
    logging.info(f'训练完成，总耗时 {(time.time() - t_start) / 60:.1f} 分钟，'
                 f'最优 val {best_val:.4f} @ 第{best_iter}步')


# ========== 训后验收 ==========
def eval_forgetting(model_path='model_best_zh_v11.pt'):
    """灾难性遗忘检查：SFT 后的模型回永久基准集打分，对比 v10 的 3.1819。

    判断口径：SFT 后 base loss 小幅变差（+0.1 以内）正常——词表扩了 3 行且分布
    偏向对话格式；但如果恶化超过 0.3，说明 LR 或 epoch 过头，得回炉调小。
    基准集文件不在本仓库（.pt 被 gitignore），缺失时先去 中文版-v10-基准测试 生成。
    """
    bench_path = '../中文版-v10-基准测试/benchmark_val.pt'
    try:
        bench = torch.load(bench_path, map_location='cpu', weights_only=True)
    except FileNotFoundError:
        raise SystemExit(f'基准集不存在: {bench_path}，请先运行 中文版-v10-基准测试/build_benchmark.py 生成')

    model = BigramLanguageModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.to(device).eval()
    logging.info(f'被测模型: {model_path}（确认路径无误再信数字，v10 误诊教训）')

    losses = torch.zeros(50)
    with torch.no_grad():
        for k in range(50):
            ix = torch.randint(len(bench) - block_size, (batch_size,))
            x = torch.stack([bench[i:i + block_size] for i in ix]).to(device)
            y = torch.stack([bench[i + 1:i + block_size + 1] for i in ix]).to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, l = model(x, y)
            losses[k] = l.item()
    print(f'{model_path}')
    print(f'基准集 val loss: {losses.mean():.4f} ± {losses.std():.4f}（v10 基线 3.1819）')


def analy_model():
    ckpt = torch.load('model_final_zh_v11.pt', map_location='cpu', weights_only=True)
    for name, tensor in ckpt.items():
        print(f'{name:45s} {tuple(tensor.shape)}')


def main():
    pass
    #train_loop()
    #chat('model_best_zh_v11.pt')
    eval_forgetting('model_best_zh_v11.pt')
    #analy_model()

if __name__ == '__main__':
    main()
