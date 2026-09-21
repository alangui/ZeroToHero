# 系统学习 LLM（大语言模型）完整路线图

---

## 一、数学基础（前置知识）

| 领域 | 核心内容 | 在 LLM 中的应用 |
|------|---------|---------------|
| **线性代数** | 矩阵运算、特征分解、SVD | Transformer 的注意力机制计算 |
| **微积分** | 梯度、偏导数、链式法则 | 反向传播、优化器 |
| **概率统计** | 条件概率、贝叶斯、最大似然估计 | 语言模型的概率建模、采样 |
| **优化理论** | 梯度下降、Adam、学习率调度 | 模型训练 |

> 📺 **推荐**：3Blue1Brown 的可视化系列（约5小时建立直觉）
> - 线性代数的本质（~3小时）
> - 微积分的本质（~2小时）
> - 概率论（~1小时）

---

## 二、深度学习基础

| 主题 | 重点 |
|------|------|
| **神经网络基础** | 前馈网络、激活函数、损失函数 |
| **反向传播** | 手动推导 + 代码实现 |
| **正则化** | Dropout、权重衰减、早停 |
| **CNN / RNN** | 理解序列建模的演进（为 Transformer 铺垫） |
| **PyTorch/TensorFlow** | 熟练搭建和训练模型 |

> 📺 **推荐**：Andrej Karpathy "Neural Networks: Zero to Hero"（从头写 GPT）
> - GitHub: https://github.com/karpathy/nn-zero-to-hero

---

## 三、Transformer 架构（LLM 的核心）

| 组件 | 必须掌握 |
|------|---------|
| **自注意力机制** | Q/K/V 计算、多头注意力、掩码注意力 |
| **位置编码** | 绝对位置编码、RoPE、ALiBi |
| **前馈网络** | FFN、激活函数（GELU/SwiGLU） |
| **层归一化** | Pre-Norm vs Post-Norm |
| **残差连接** | 梯度流动、训练稳定性 |
| **Tokenizer** | BPE、WordPiece、SentencePiece |

> 📖 **必读论文**：《Attention Is All You Need》(2017)

---

## 四、预训练（Pre-training）

| 主题 | 内容 |
|------|------|
| **数据工程** | 数据清洗、去重、质量过滤、数据配比 |
| **训练目标** | 自回归语言建模（Next Token Prediction） |
| **大规模训练** | 分布式训练（DDP、FSDP、DeepSpeed、Megatron-LM） |
| **混合精度** | FP16/BF16、梯度缩放 |
| **长上下文** | 位置编码外推、Ring Attention |

---

## 五、微调与对齐（Fine-tuning & Alignment）

| 技术 | 说明 |
|------|------|
| **全参数微调 (SFT)** | 在领域数据上继续训练 |
| **参数高效微调 (PEFT)** | LoRA、QLoRA、Adapter、Prefix Tuning |
| **指令微调** | 构建指令数据集、对话模板 |
| **RLHF** | 奖励模型 + PPO 强化学习 |
| **DPO/RLAIF** | 更简单的对齐方法（替代 RLHF） |

---

## 六、推理与部署（Inference & Deployment）

| 主题 | 技术 |
|------|------|
| **推理优化** | KV Cache、量化（INT8/INT4/AWQ/GPTQ）、投机解码 |
| **服务框架** | vLLM、TensorRT-LLM、TGI、SGLang |
| **长文本推理** | 滑动窗口、稀疏注意力 |
| **批处理策略** | Continuous Batching、Dynamic Batching |

---

## 七、应用开发（LLM Application）

| 方向 | 技术栈 |
|------|--------|
| **Prompt Engineering** | 零样本/少样本、CoT、ReAct、Self-Consistency |
| **RAG** | 向量数据库（Milvus/Pinecone）、检索策略、重排序 |
| **Agent** | 工具调用、多 Agent 协作、记忆管理 |
| **框架** | LangChain、LlamaIndex、Dify、AutoGen |

---

## 八、评估与安全（Evaluation & Safety）

| 主题 | 内容 |
|------|------|
| **评估指标** | Perplexity、BLEU、ROUGE、MMLU、HumanEval |
| **基准测试** | 通用能力、代码、数学、多语言 |
| **幻觉检测** | 事实性、一致性、引用溯源 |
| **安全对齐** | 越狱攻击、偏见、隐私保护、红队测试 |

---

## 九、LLMOps（工程化）

| 环节 | 工具/实践 |
|------|----------|
| **实验管理** | W&B、MLflow、TensorBoard |
| **模型版本** | Hugging Face Hub、Model Registry |
| **监控** | 延迟、吞吐量、token 消耗、用户反馈 |
| **CI/CD** | 自动化训练、评估、部署流水线 |
| **成本优化** | 模型蒸馏、MoE、动态路由 |

---

## 📋 推荐学习路线（3-6个月）

### 阶段1：基础（4-6周）
- 数学复习（线性代数 + 微积分 + 概率）
- 深度学习基础（PyTorch + 神经网络）
- 手写一个简单 Transformer

### 阶段2：核心（4-6周）
- 精读 Attention Is All You Need
- 用 Hugging Face 预训练一个小模型
- 实践 LoRA 微调
- 学习 RLHF/DPO 原理

### 阶段3：应用（4-6周）
- Prompt Engineering 系统学习
- 搭建一个 RAG 系统
- 开发一个简单 Agent
- 学习 vLLM 部署

### 阶段4：深入（持续）
- 阅读最新论文（arXiv daily）
- 参与开源项目（vLLM、Transformers）
- 专精一个方向（预训练/对齐/推理优化/Agent）

---

## 📚 核心资源汇总

| 类型 | 资源 |
|------|------|
| **课程** | DeepLearning.AI Generative AI with LLMs、Hugging Face LLM Course |
| **书籍** | 《动手学深度学习》（李沐）、《Building LLMs from Scratch》（Raschka） |
| **论文** | Attention Is All You Need、GPT-3、LLaMA、DPO |
| **代码** | nanoGPT（Karpathy）、Transformers 库源码 |
| **社区** | Hugging Face、Papers With Code、Reddit r/LocalLLaMA |

---

*整理时间：2026-09-21*
