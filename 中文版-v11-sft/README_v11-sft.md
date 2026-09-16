# 中文版-v11-sft：v10 基座 + 指令微调（SFT）→ 可对话的 GPT 助手

- 定位：**相对 v10 的唯一变量 = +SFT**（指令数据 + 词表扩维 + loss mask）
- 基座：`../中文版-v10-gpu/model_best_zh_v10.pt`（234.5M，永久基准 3.1819）
- 模型骨架与 v8~v10 完全一致，唯一结构差异：词表 16000 → 16003

## 文件清单

| 文件 | 说明 |
|---|---|
| `sft_zh_v11.py` | 训练 + 对话推理 + 基准补测单文件（项目惯例，main() 注释切换入口） |
| `prepare_sft_data.py` | 指令数据 → tokenized 样本（ids + labels），支持 alpaca/Belle 单轮、ShareGPT 多轮、整文件 JSON 数组与 JSONL |
| `download_sft_data.py` | 下载/导入指令数据并合并去重（alpaca-gpt4-zh 自动下载，Belle 传本地路径） |
| `demo_data.jsonl` | 17 条手写中文指令，用于不调数据先跑通管线 |
| `zh_bpe.model` / `zh_bpe.vocab` | 与 v10 完全相同的 BPE 分词模型（原样复制） |
| `train.log` | 训练日志（跑训练后生成） |

## 对话格式与词表扩维（核心设计）

特殊符号不放 sentencepiece 词表（它没有空闲 id，且 `sp.decode` 对越界 id 直接抛错），而是放到扩展区：

| 符号 | id | 作用 |
|---|---|---|
| `<|user|>` | 16000 | 用户问题开始 |
| `<|assistant|>` | 16001 | 助手回答开始 |
| `<|eot|>` | 16002 | 一轮对话结束（模型学会它才会"收口"） |

训练样本格式（多轮时每个回答都算 loss）：

```
ids    = [USER] + 问题 + [ASSISTANT] + 回答 + [EOT]（多轮则继续拼下一轮）
labels = [-100] 屏蔽用户部分，只在 回答 + [EOT] 上计算 loss
```

模型侧：`token_embd` 和 `lm_head` 各扩 3 行（v7 起无 tying，两份都要动），
前 16000 行从 v10 checkpoint 原样拷贝，新行 N(0, 0.02) 初始化。
推理时 `decode_ids()` 先过滤特殊符号再解码。

## 三步走

### 1. 准备指令数据

量级建议：3~5 万条起步，234M 基座吃不多。开源中文指令数据集：

- **Belle**（35 万+，[LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)）
- **alpaca-gpt4-data-zh**（5 万，[Instruction-Tuning-with-GPT-4/GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)）
- **Firefly-train-1.1M**（含多轮，[YeungNLP/Firefly](https://github.com/YeungNLP/Firefly)）

`download_sft_data.py` 负责下载和合并（alpaca-gpt4-zh 自动从 GitHub 下载；
Belle 文件 600MB+ 需手动从 HuggingFace 下载后传路径）：

```bash
# 只用它：自动下载 alpaca-gpt4-zh（43MB）→ 去重采样 → sft_all.jsonl
python download_sft_data.py

# 合并 Belle
python download_sft_data.py --belle path/to/train_0.5M_CN.json

# 追加其他同格式文件、调整采样量
python download_sft_data.py --extra other.jsonl --max-samples 80000
```

然后 tokenize（超过 256 token 的样本直接丢弃，这是 block_size 的硬约束）：

```bash
# 先拿 demo 数据跑通管线（17 条，只验证格式不进正式训练）
python prepare_sft_data.py --demo

# 正式数据
python prepare_sft_data.py --input sft_all.jsonl --output sft_data.pt
```

**实测数据**（2026-09-16）：alpaca-gpt4-zh 48818 条 → 可用 32697 条，
**33% 因超 256 token 被丢弃**（该数据集回答偏长，属预期）；全量 tokenize 耗时 8 秒。
Belle 回答普遍更短，丢弃率会低不少，两者混用能补回量。

### 2. 训练（在 GPU 笔记本上）

```bash
cd 中文版-v11-sft && python sft_zh_v11.py   # main() 里取消 train_loop() 的注释
```

| 超参 | v10 预训练 | v11 SFT | 理由 |
|---|---|---|---|
| LR | 3e-4 → 3e-5 | **2e-5 → 1e-6** | 基座已训好，LR 低一个数量级防遗忘 |
| batch | 32 | 16 | 样本短且变长，padding 后批内差异大 |
| dropout | 0.2 | **0.1** | SFT 目标是记住格式+迁移知识 |
| 梯度检查点 | 开 | **关** | 样本短显存压力小，省 30% 速度损耗 |
| best 保存门槛 | iter > 20000 | 无门槛 | SFT 总共几千步 |

### 3. 训后验收（顺序不能乱）

```bash
# ① 格式验收：会不会自己用 <|eot|> 收口（不收口 = loss mask 或 id 拼接错了）
#    main() 里取消 chat() 注释；多轮对话，exit 退出

# ② 遗忘检查：回永久基准集打分，对比 v10 的 3.1819
#    恶化 < +0.1 正常；> +0.3 说明 LR/epoch 过头，回炉调小
python sft_zh_v11.py   # main() 里取消 eval_forgetting() 注释
```

然后按项目惯例写 `v11-sft模型训练总结.md`（数据构成、唯一变量、loss 记录、
chat 盲测结论、基准补测数字），更新全版本总览表。

## 已知坑

1. **`<|eot|>` 学不出来**：新符号行是小初始化，前几百步 loss 偏高正常；训完仍不收口先查 mask 和 id 拼接，别急着调超参
2. **灾难性遗忘**：LR 超 1e-4 或 epoch 超 3，wiki 知识肉眼可见退化——②号验收就是抓这个的
3. **block_size=256 截断**：长回答被切断会让模型学到"说到一半就停"，验收时专门看长回答；如需多轮长对话，扩 512 + position_embd 插值是后续消融方向
4. **基准集文件不在仓库**：`benchmark_val.pt` 被 .gitignore 排除，缺失时先去 `中文版-v10-基准测试/` 跑 `build_benchmark.py`
5. **评估时核对被测模型路径**：v10 误诊教训——eval 输出首行打印模型路径，确认是本人再信数字
