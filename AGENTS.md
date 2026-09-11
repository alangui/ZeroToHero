# AGENTS.md

> 本文件面向 AI 编码代理（Agent），介绍本项目的结构、约定与注意事项。阅读前假设你对本项目一无所知。

## 一、项目概述

这是一个**个人学习/实验项目**，跟随 Andrej Karpathy 的 "Zero to Hero" 系列，**从零手写 GPT（decoder-only Transformer）语言模型**，并逐步迭代训练中文 GPT。项目不是库也不是服务，没有打包/发布流程，全部代码都是**可直接运行的单文件训练/推理脚本**。

- 语言：Python 3，核心框架 **PyTorch**
- 注释、日志、训练总结文档全部使用**中文**，新增代码和文档请继续使用中文
- 本地开发机：MacBook Pro（Apple Silicon，MPS 后端）；GPU 训练机：RTX 4060 Laptop（8GB 显存，CUDA）

## 二、目录结构与模块划分

```
ZeroToHero/
├── gpt.py                    # 英文版起点：tiny-shakespeare 字符级 GPT（Karpathy 原始教程的中文注释+扩展版）
├── gpt_zh.py                 # 中文版：AMC 语料字符级 GPT，增加 loss 曲线图/JSON 记录
├── prepare_data.py           # 语料预处理：合并 AMC_mini 四个子语料 → 白名单清洗 → corpus_clean_zh.txt + vocab_zh.json
├── gpt2/train_gpt2_1.py      # 未完成的残稿（只有 4 行，勿当作可运行代码）
├── tiny-shakespeare.txt      # 英文训练语料
├── corpus_clean_zh.txt       # 清洗合并后的中文语料（prepare_data.py 产物，~55MB，Git LFS 管理）
├── vocab_zh.json             # 字符级词表（prepare_data.py 产物）
├── model_final*.pt           # 模型权重（*.pt 已被 .gitignore 排除）
├── AMC_mini/                 # Alpha 现代汉语语料库子集：对话/小说/非虚构/科技 各 100K 篇 + 官方文档（docx）
└── 中文版-v*/                 # 版本化实验目录，每个版本一个文件夹（见下）
```

### 版本化实验目录（中文版-v1 ~ v10-gpu）

每个 `中文版-vN` 目录是一次**自包含的完整实验**，典型内容：

| 文件 | 说明 |
|---|---|
| `gpt_zh_vN(-gpu).py` | 该版本的训练+推理脚本（单文件，模型定义、训练循环、推理函数都在里面） |
| `vN模型训练总结.md` | **必读**：该版本的训练配置、loss 记录、结论与下版本计划 |
| `loss_curve_vN.png` / `losses_record_vN.json` | loss 曲线图与原始数据 |
| `train.log` | GPU 版训练日志（logging 模块同时写文件和终端） |
| `zh_bpe.model` / `zh_bpe.vocab` | v5 起使用的 sentencepiece BPE 分词模型（vocab 16000） |
| `clean_chars.py` | 语料白名单清洗脚本 |

- `中文版-v7-gpu/` 内含 `中文版-v7.1-gpu/`、`中文版-v7.2-gpu/` 两个**消融实验**子目录（用于定位 tying 权重共享的负面影响）

### 版本演进主线（各版本总结文档中有完整对比表）

1. **v1~v4**：字符级分词，MPS → CUDA，模型从 31M 扩到 96M
2. **v5~v6**：切换到 sentencepiece **BPE 分词**（vocab 16000，`[BR]` 自定义符号保留换行）
3. **v7 / v7.1 / v7.2**：消融实验，定罪 embedding/lm_head 权重共享（tying）有害，**永久移除**
4. **v8~v9**：干净配方长训（234.5M 参数，16 层 × 1024 维 × 16 头），余弦学习率 + best-checkpoint 保存，v9 探明 0.1b 语料的数据天花板（best val 3.4611）
5. **v10**：扩语料到 `wiki_corpus_0.3b_clean.txt`

## 三、技术栈与运行方式

### 依赖

无 `requirements.txt` / `pyproject.toml`。Python 环境用 **conda** 管理（见 `.vscode/settings.json`）。实际依赖：

- `torch`（MPS 或 CUDA 版）
- `matplotlib`（loss 曲线，脚本内已配置中文字体防乱码）
- `sentencepiece`（v5 起的 BPE 分词）
- `numpy`

### 运行

没有构建步骤，直接运行脚本。**注意工作目录**：脚本内的相对路径（语料、BPE 模型、输出权重）均相对于脚本所在目录，因此应 `cd` 进对应版本目录再运行：

```bash
cd 中文版-v10-gpu && python gpt_zh_v10-gpu.py
```

每个脚本的 `main()` 里用**注释切换入口函数**（`train_loop()` / `load_model_for_inference()` / `load_mode_generate_txt_streaming()` / `analy_model()`），要跑哪个就取消哪个的注释——这是本项目的固定用法，改动时保持这个模式。

### 模型架构（所有版本共用同一骨架）

单文件内依次为：`Head`（单头因果注意力）→ `MultiHeadAttention` → `FeedFoward`（注意拼写就是 FeedFoward，勿"修正"）→ `Block`（pre-norm 残差块）→ `BigramLanguageModel`（类名是历史遗留，实为完整 GPT：token/位置嵌入 + N 个 Block + LayerNorm + lm_head）。超参数全部是**文件顶部模块级常量**（`batch_size`、`block_size`、`n_embd`、`n_head`、`n_layer`、`learning_rate` 等）。

GPU 版的关键优化（v4 起逐步加入，改动新版时应继承）：

- `F.scaled_dot_product_attention(is_causal=True)`（Flash Attention，不物化注意力矩阵）
- `torch.utils.checkpoint` 梯度检查点（省 60%+ 激活显存，代价约 30% 训练速度）
- fp16 `autocast` + `GradScaler` 混合精度
- 余弦学习率调度（3e-4 → 3e-5，100 步 warmup），AdamW（wd 0.1）
- best checkpoint 机制：val loss 创新低且 `iter > 阈值`（最后 1/4 进程）时保存 `model_best_zh_vN.pt`；推理务必用 best 而非 final

### 数据与分词约定

- **清洗**：白名单制（`keep_char()`）——基本汉字、扩展A区汉字、中文标点、换行、数字，其余一律删除；`\n{3,}` 压缩为 `\n\n`
- **BPE 换行坑**（v5 总结中的教训）：sentencepiece 按行训练，`\n` 永远不会成为 token。必须注册 `user_defined_symbols=['[BR]']`，编码时逐行 encode 并在行间插入 `[BR]` 的 id，解码后 `.replace('[BR]', '\n')`
- **训练/验证切分**：GPU 版用 `random_split()` 按 1024-token chunk 随机切分（seed 1337），保证跨版本 val 可比；换语料后 val 不可比，需用旧 val 集补测
- **wiki 大语料**（`wiki_corpus_0.*b*.txt`）不进版本控制（.gitignore），需要时由各版本目录下的 `clean_chars.py` 从原始语料清洗生成

## 四、实验方法论与开发约定

- **单变量原则**：每版只改一个变量，总结文档中明确写出"相对上版的唯一改动"；v7 因 5 变量混杂成为负面教材
- **每版必写训练总结** `vN模型训练总结.md`：含硬件、完整超参数表、loss 全记录、与上版同 val 集对比、结论、下版本计划；总结末尾维护**全版本总览表**
- **预登记**：新版本开训前在上一版总结中写下预测指标，训完对照验收
- 代码注释解释"为什么这么做"（如显存优化的取舍），风格保持中文、口语化、带具体数字

## 五、测试与版本控制

- **没有自动化测试套件**。验证方式是：`analy_model()` 打印权重形状做 sanity check，以及推理生成文本人工检查质量；`中文版-v5-gpu/test.ipynb` 是随手实验的 notebook，不是测试
- Git 约定：`*.pt` 权重不入库；`*.txt` 走 **Git LFS**（见 `.gitattributes`）；超大 wiki 语料被 `.gitignore` 排除
- 无 CI/CD、无部署流程

## 六、注意事项

- 加载旧版 `model_final.pt`（完整模型 pickle）需要 `weights_only=False`；新版都是 `state_dict`
- GPU 脚本默认 `device='cuda'`，在 Mac 上跑需改回 `'mps'`（或参考根目录脚本的自动检测写法）
- 8GB 显存是硬约束：增大 batch/模型前先看对应版本总结中的显存峰值记录
