# -*- coding: utf-8 -*-
# prepare_sft_data.py —— 指令数据 → tokenized SFT 训练样本
#
# 把开源指令数据集（alpaca / Belle 格式，或 ShareGPT 多轮格式）转成模型能吃的形式：
#   ids    : [USER_ID] + encode(问题) + [ASSISTANT_ID] + encode(回答) + [EOT_ID] (+ 多轮继续)
#   labels : 用户部分全部 -100（不计算 loss），只在回答部分和 [EOT_ID] 上计算 loss
#
# 特殊符号约定（与 sft_zh_v11.py 严格一致，改一处必须同步改另一处）：
#   词表是 sentencepiece 的 16000 个 BPE piece，没有空闲 id 可复用，
#   所以 USER/ASSISTANT/EOT 三个符号直接放到 16000~16002 的扩展区，
#   模型 embedding 也会相应扩到 16003（见 sft_zh_v11.py 的词表扩维手术）。
#   注意：sp.decode 对越界 id 会抛 IndexError，推理解码时必须过滤，训练不受影响。
import argparse
import json
import torch
import sentencepiece as spm

SP_VOCAB_SIZE = 16000      # zh_bpe.model 的原生词表大小，不要改
NL_ID = 3                  # [BR] 换行符号的 id（zh_bpe.model 里实测为 3）
USER_ID = 16000            # <|user|>
ASSISTANT_ID = 16001       # <|assistant|>
EOT_ID = 16002             # <|eot|> 一轮对话结束，模型学会它才会"收口"

sp = spm.SentencePieceProcessor(model_file='zh_bpe.model')


def encode_text(t):
    """与预训练完全一致的编码方式：逐行 encode，行间插入 [BR]，保持换行习惯不漂移"""
    return [i for line in sp.encode(t.split('\n')) for i in line + [NL_ID]]


def build_sample(turns):
    """turns: [(question, answer), ...]。返回 (ids, labels)；多轮时每个回答都参与 loss。

    设计取舍说明：多轮样本只在 assistant 回答（含 [EOT]）上算 loss，用户的问题不算。
    全部轮次的回答都算 loss（比只算最后一轮更充分利用数据），代价是同一文本以不同
    前缀重复出现，样本间有相关性——数据量不大时这是可接受的交换。
    """
    ids, labels = [], []
    for q, a in turns:
        user_part = [USER_ID] + encode_text(q) + [ASSISTANT_ID]
        ans_part = encode_text(a) + [EOT_ID]
        ids += user_part + ans_part
        labels += [-100] * len(user_part) + ans_part
    return ids, labels


def parse_record(rec):
    """把一条原始数据统一解析成 [(q, a), ...]。支持两种格式：

    1. alpaca / Belle 单轮: {"instruction": ..., "input": ..., "output": ...}
       input 非空时拼到 instruction 后面（Belle 的常用套路）
    2. ShareGPT 多轮: {"conversations": [{"from": "human"/"gpt", "value": ...}, ...]}
       按 human/gpt 配对；配对不齐（比如最后是 human 没回答）就丢弃
    """
    if 'conversations' in rec:
        turns, q, a = [], None, None
        for msg in rec['conversations']:
            role, v = msg.get('from'), msg.get('value', '')
            if role in ('human', 'user'):
                if q is not None and a is not None:
                    turns.append((q, a)); q, a = None, None
                q = v
            elif role in ('gpt', 'assistant'):
                a = v
        if q is not None and a is not None:
            turns.append((q, a))
        return turns if turns else None

    inst, inp, out = rec.get('instruction'), rec.get('input'), rec.get('output')
    if not inst or not out:
        return None
    q = inst + ('\n' + inp if inp else '')
    return [(q, out)]


def iter_records(path):
    """兼容整文件 JSON 数组与 JSONL 两种存储格式。"""
    with open(path, 'r', encoding='utf-8') as f:
        if f.read(1) == '[':
            f.seek(0)
            yield from json.load(f)
            return
        f.seek(0)
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='demo_data.jsonl', help='原始指令数据 jsonl 路径')
    ap.add_argument('--output', default='sft_data.pt', help='输出文件（torch.save 格式）')
    ap.add_argument('--max-len', type=int, default=256, help='样本最大 token 数（超过直接丢弃）')
    ap.add_argument('--demo', action='store_true', help='只处理内置 demo 数据（等价于 --input demo_data.jsonl）')
    args = ap.parse_args()

    src = 'demo_data.jsonl' if args.demo else args.input
    samples, n_skip_long, n_skip_bad = [], 0, 0
    for rec in iter_records(src):
        try:
            turns = parse_record(rec)
        except (AttributeError, TypeError, KeyError):
            n_skip_bad += 1
            continue
        if not turns:
            n_skip_bad += 1
            continue
        ids, labels = build_sample(turns)
        if len(ids) > args.max_len:
            n_skip_long += 1
            continue
        samples.append({'ids': ids, 'labels': labels})

    if not samples:
        raise SystemExit(f'没有可用样本！请检查 {src} 的格式（支持 alpaca / ShareGPT 两种）')

    torch.save(samples, args.output)
    total = len(samples)
    ans_tokens = sum(sum(1 for x in s['labels'] if x != -100) for s in samples)
    all_tokens = sum(len(s['ids']) for s in samples)
    print(f'来源: {src}')
    print(f'可用样本 {total} 条，丢弃超长 {n_skip_long} 条，格式非法 {n_skip_bad} 条')
    print(f'总 token {all_tokens:,}，其中回答部分（参与 loss）{ans_tokens:,} '
          f'({ans_tokens / all_tokens:.1%})')
    print(f'样本平均长度 {all_tokens / total:.0f} token，最长 {max(len(s["ids"]) for s in samples)}')
    print(f'已保存到 {args.output}')


if __name__ == '__main__':
    main()
