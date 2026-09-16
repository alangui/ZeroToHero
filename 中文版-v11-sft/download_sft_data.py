# -*- coding: utf-8 -*-
# download_sft_data.py —— 下载/导入中文指令数据，合并去重成一份 sft_all.jsonl
#
# 用法：
#   python download_sft_data.py                          # 只下 alpaca-gpt4-zh（约 43MB，GitHub 直链）
#   python download_sft_data.py --belle path/to/belle.jsonl   # 合并 Belle（600MB+，需手动下载，见下）
#   python download_sft_data.py --belle ... --max-samples 50000
#
# 数据源：
#   1. alpaca-gpt4-data-zh（5 万条，单轮，GPT-4 翻译）—— 脚本自动下载
#      来源：https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
#   2. Belle train_0.5M_CN（50 万条，单轮，量大质量参差）—— 手动下载后传路径
#      https://huggingface.co/datasets/BelleGroup/train_0.5M_CN
#      （文件 600MB+，HuggingFace 下载比 GitHub 稳；下不动就只用 alpaca-gpt4-zh 也够跑通）
#   3. 其他任意同格式 jsonl 用 --extra 追加
#
# 合并后用 prepare_sft_data.py 做 tokenize：
#   python prepare_sft_data.py --input sft_all.jsonl --output sft_data.pt
import argparse
import hashlib
import json
import random
import sys
import urllib.request

ALPACA_GPT4_ZH_URL = 'https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/main/data/alpaca_gpt4_data_zh.json'
ALPACA_LOCAL = 'alpaca_gpt4_data_zh.json'


def download_alpaca():
    print(f'下载 alpaca-gpt4-data-zh ...')
    print(f'  {ALPACA_GPT4_ZH_URL}')
    try:
        urllib.request.urlretrieve(ALPACA_GPT4_ZH_URL, ALPACA_LOCAL)
    except Exception as e:
        sys.exit(f'下载失败: {e}\n'
                 f'手动下载上面的链接放到本目录，再重新运行本脚本（检测到文件会自动跳过下载）')
    print('  完成')


def iter_records(path):
    """兼容三种存储格式：整文件 JSON 数组、JSONL、Belle 的 {键: 数组} 字典。"""
    with open(path, 'r', encoding='utf-8') as f:
        head = f.read(1)
        f.seek(0)
        if head == '[':
            yield from json.load(f)
            return
        first_line = f.readline()
        try:
            rec = json.loads(first_line)
            if isinstance(rec, dict) and not ('instruction' in rec or 'conversations' in rec):
                # Belle 风格：{instruction: [...], output: [...]} 按位置配对
                keys = list(rec.keys())
                n = len(rec[keys[0]])
                for i in range(n):
                    yield {k: rec[k][i] for k in keys}
                return
            yield rec
        except json.JSONDecodeError:
            return
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize(rec):
    """统一成 alpaca 格式。过滤掉明显不合格的样本（空字段/过短回答）。"""
    if 'conversations' in rec:
        return rec  # ShareGPT 多轮，prepare 脚本原生支持
    inst, inp, out = rec.get('instruction'), rec.get('input'), rec.get('output')
    if not inst or not out or len(out.strip()) < 10:
        return None
    return {'instruction': inst.strip(), 'input': (inp or '').strip(), 'output': out.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--belle', help='Belle train_0.5M_CN 本地文件路径（jsonl 或 json 数组均可）')
    ap.add_argument('--extra', action='append', default=[], help='追加其他同格式数据文件，可多次指定')
    ap.add_argument('--max-samples', type=int, default=50000, help='合并去重后的采样上限（默认 5 万）')
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    import os
    if not os.path.exists(ALPACA_LOCAL):
        download_alpaca()

    sources = [(ALPACA_LOCAL, 'alpaca-gpt4-zh')]
    if args.belle:
        sources.append((args.belle, 'Belle'))
    for p in args.extra:
        sources.append((p, p))

    seen, merged, stats = set(), [], {}
    for path, name in sources:
        n_new, n_skip = 0, 0
        for rec in iter_records(path):
            rec = normalize(rec)
            if rec is None:
                n_skip += 1
                continue
            key = hashlib.md5(json.dumps(rec, sort_keys=True, ensure_ascii=False)
                              .encode('utf-8')).hexdigest()
            if key in seen:
                n_skip += 1
                continue
            seen.add(key)
            merged.append(rec)
            n_new += 1
        stats[name] = (n_new, n_skip)
        print(f'{name}: 新增 {n_new} 条，去重/过滤 {n_skip} 条')

    random.seed(args.seed)
    random.shuffle(merged)
    if len(merged) > args.max_samples:
        print(f'随机采样 {args.max_samples} / {len(merged)} 条（--max-samples 可调整）')
        merged = merged[:args.max_samples]

    out = 'sft_all.jsonl'
    with open(out, 'w', encoding='utf-8') as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'\n已写入 {out}，共 {len(merged)} 条')
    print('下一步: python prepare_sft_data.py --input sft_all.jsonl --output sft_data.pt')


if __name__ == '__main__':
    main()
