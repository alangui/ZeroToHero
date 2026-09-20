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
#      实际文件名 Belle_open_source_0.5M.json（286MB），页面点 Files → 下载；
#      国内网络打不开就把 huggingface.co 换成镜像 hf-mirror.com，路径不变
#      ⚠️ 许可限制：GPL-3.0，且官方声明仅限研究用途、不得商用
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
IDENTITY_LOCAL = 'identity_data.jsonl'   # 手写身份/能力问答，防"你是谁"类问题冷场（v11 盲测教训）


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


# 2026-09-19 v11 盲测教训：Belle 的拒答模板和输入复述任务会直接训出"抱歉怪+复读机"
# （见 v11-sft模型训练总结 2.6 节），合并时统一拦截
# 2026-09-20 v11.3 盲测教训：拒答开场白变体极多（"很抱歉/我非常抱歉/非常抱歉"），
# startswith 玩前缀是打地鼠——改用"前 8 字包含即拦截"，变体一网打尽
REFUSAL_PREFIXES = ('我无法', '我不能', '作为一个')


def is_refusal(out):
    """拒答开场白检测：前 8 个字里出现"抱歉/对不起"即判定（覆盖 很抱歉/非常抱歉/
    我很抱歉 等所有语序变体），或以"我无法/我不能/作为一个"开头。"""
    return ('抱歉' in out[:8] or '对不起' in out[:8]
            or out.startswith(REFUSAL_PREFIXES))
ECHO_KEYWORDS = ('重复', '倒序', '倒过来', '逆向', '反过来')


def normalize(rec):
    """统一成 alpaca 格式。过滤明显不合格的样本（空字段/过短回答/拒答模板/复述任务）。"""
    if 'conversations' in rec:
        return rec  # ShareGPT 多轮，prepare 脚本原生支持
    inst, inp, out = rec.get('instruction'), rec.get('input'), rec.get('output')
    if not inst or not out or len(out.strip()) < 10:
        return None
    out = out.strip()
    # 拒答模板：Belle/alpaca 成批的"（我）（很/非常）抱歉…"，会学成全场景兜底话术
    if is_refusal(out):
        return None
    # 输入复述任务：Belle 大量"重复输入/倒序/改写"类，回答是输入的拷贝，
    # 训出的默认策略是复读用户的话
    if any(k in inst for k in ECHO_KEYWORDS):
        return None
    return {'instruction': inst.strip(), 'input': (inp or '').strip(), 'output': out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--belle', help='Belle train_0.5M_CN 本地文件路径（jsonl 或 json 数组均可）')
    ap.add_argument('--extra', action='append', default=[], help='追加其他同格式数据文件，可多次指定')
    ap.add_argument('--max-samples', type=int, default=50000, help='合并去重后的采样上限（默认 5 万）')
    ap.add_argument('--belle-cap', type=int, default=15000,
                    help='Belle 降采样上限（默认 1.5 万；v11.2 教训：Belle 占 91% 会把模型 '
                         '训成模板文体且对口语短问题 OOD 坍缩，压到 ~25% 配比）')
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    import os
    if not os.path.exists(ALPACA_LOCAL):
        download_alpaca()

    sources = [(ALPACA_LOCAL, 'alpaca-gpt4-zh'), (IDENTITY_LOCAL, 'identity(手写身份)')]
    if args.belle:
        sources.append((args.belle, 'Belle'))
    for p in args.extra:
        sources.append((p, p))

    seen, merged, stats = set(), [], {}
    for path, name in sources:
        n_new, n_skip = 0, 0
        cap = args.belle_cap if name == 'Belle' else None   # Belle 降采样（v11.3 药方）
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
            if cap is not None and n_new >= cap:
                n_skip += 1
                continue
            merged.append(rec)
            n_new += 1
        stats[name] = (n_new, n_skip)
        print(f'{name}: 新增 {n_new} 条，去重/过滤/降采样 {n_skip} 条')

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
