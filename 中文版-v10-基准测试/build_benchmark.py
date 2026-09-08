# build_benchmark.py —— 建永久基准验证集
# 输入: benchmark_source_clean.txt（已过 clean_chars 的维基文本，可用 0.4b 清洗版）
# 输出: benchmark_val.txt（文本备查）+ benchmark_val.pt（冻结的 token 张量）
import torch
import sentencepiece as spm
import numpy as np

# 1. 载入 0.3b 的所有行（0.3b 是所有历史模型见过的最大语料，0.1b 是它的子集）
train_lines = set()
with open('../wiki_corpus_0.3b_clean.txt', encoding='utf-8') as f:
    for line in f:
        train_lines.add(line.rstrip('\n'))
print(f'0.3b 去重行数: {len(train_lines):,}')

# 2. 候选文本剔除重叠行
keep, dropped = [], 0
with open('../wiki_corpus_0.4b_clean.txt', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line:
            continue
        if line in train_lines:
            dropped += 1
        else:
            keep.append(line)
print(f'候选 {len(keep) + dropped:,} 行, 剔除重叠 {dropped:,} 行, 保留 {len(keep):,} 行')

with open('benchmark_val.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(keep))

# 3. 编码并冻结（zh_bpe.model 在子文件夹里的话，改成对应路径，如 中文版-v9-gpu/zh_bpe.model）
sp = spm.SentencePieceProcessor(model_file='zh_bpe.model')
NL_ID = sp.piece_to_id('[BR]')
ids = np.concatenate([np.array(l + [NL_ID], dtype=np.int32) for l in sp.encode(keep)])
torch.save(torch.from_numpy(ids.astype(np.int64)), 'benchmark_val.pt')
print(f'基准集冻结完成: {len(ids):,} token')  # 有 200 万以上就够（50 轮评估只消耗 41 万）