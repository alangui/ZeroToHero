# -*- coding: utf-8 -*-
# diagnose_chat.py —— 鉴别诊断：模型是"没学会按问题回答"还是"泛化差"
#
# 做法：从训练集抽 N 条样本，把其中的问题原样喂回模型（训练分布内），
#       生成回答与样本里的标准答案对照。
#   - 原题都答非所问 → 训练/推理格式有 bug，深挖管线
#   - 原题像样、新题拉胯 → 数据配方问题（Belle 占比/质量），治本在数据
#
# 运行：python diagnose_chat.py（在本目录，需要 model_best_zh_v11.pt 和 sft_data.pt）
import random
import torch
import importlib.util

spec = importlib.util.spec_from_file_location('sft', 'sft_zh_v11.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

N = 5
SEED = 42


def recover_qa(sample):
    """从 token 序列还原 (问题, 标准回答) 文本。"""
    ids, labels = sample['ids'], sample['labels']
    # 问题 = USER_ID 之后到 ASSISTANT_ID 之前
    u_end = ids.index(m.ASSISTANT_ID)
    q = m.decode_ids(ids[1:u_end])
    # 标准回答 = labels 非 -100 部分（含 EOT，解码时自动过滤）
    ref = m.decode_ids([t for t, l in zip(ids, labels) if l != -100])
    return q.strip(), ref.strip()


data = torch.load('sft_data.pt', weights_only=False)
random.seed(SEED)
picks = random.sample(data, N)

model = m.BigramLanguageModel()
model.load_state_dict(torch.load('model_best_zh_v11.pt', map_location='cpu', weights_only=True))
model.to(m.device).eval()

for i, s in enumerate(picks):
    q, ref = recover_qa(s)
    ctx = m.build_chat_context([], q)
    gen = m.generate_chat(model, ctx)
    print(f'===== 样本 {i} =====')
    print(f'[问题] {q[:80]}')
    print(f'[标准答案] {ref[:120]}')
    print(f'[模型回答] {gen[:120]}')
    print()
