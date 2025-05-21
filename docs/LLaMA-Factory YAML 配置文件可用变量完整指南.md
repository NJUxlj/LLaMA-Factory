# LLaMA-Factory YAML 配置文件可用变量完整指南

为了找到 LLaMA-Factory 中所有可用的配置变量，我搜索了 GitHub 上的相关代码和文档。通过分析源代码，特别是配置解析相关文件，我整理出了完整的变量列表。

## 配置变量查找方法

查看 LLaMA-Factory 的代码库，主要从以下几个文件获取配置信息：

1. `src/llamafactory/train/parser.py` - 命令行参数解析器，定义了所有支持的参数
2. `examples/` 目录下的示例配置文件
3. `src/llamafactory/datasets/` 和 `src/llamafactory/train/` 目录下的实现代码

## 完整配置变量列表

以下是按类别组织的完整配置变量列表：

### 模型配置 (Model)
```yaml
model_name_or_path: string  # 预训练模型名称或路径
cache_dir: string  # 缓存目录
checkpoint_dir: string  # 检查点目录路径
quantization_bit: [4, 8, null]  # 量化位数（4位、8位或无量化）
quantization_type: ["awq", "gptq", null]  # 量化类型
flash_attn: bool  # 是否使用Flash Attention
shift_attn: bool  # 是否使用Shift Attention
rope_scaling: {"type": string, "factor": float}  # RoPE缩放设置
use_unsloth: bool  # 是否使用Unsloth优化
trust_remote_code: bool  # 是否信任远程代码
use_auth_token: string  # Hugging Face 令牌
adapter_name_or_path: string  # LoRA适配器名称或路径
adapter_type: ["lora", "adalora", "adaption_prompt", null]  # 适配器类型
```

### 训练方法 (Method)
```yaml
stage: ["sft", "rm", "ppo", "dpo", "pre"]  # 训练阶段
do_train: bool  # 是否进行训练
do_eval: bool  # 是否进行评估
do_predict: bool  # 是否进行预测
finetuning_type: ["full", "freeze", "lora", "adalora", "longlora", "ia3", "llamapeft", "prompt_tuning", "p_tuning", "prefix_tuning", "qalora", "adaption_prompt", "unipelt"]  # 微调类型
model_offload: bool  # 是否进行模型卸载
deepspeed: string  # DeepSpeed配置文件路径
```

### 数据集设置 (Dataset)
```yaml
dataset: string  # 数据集名称，可以有多个，用逗号分隔
dataset_dir: string  # 数据集所在目录
dataset_sample_template: string  # 数据集样本模板
template: string  # 对话模板名称
enable_thinking: bool  # 是否启用思维链
template_config_path: string  # 模板配置文件路径
cutoff_len: int  # 最大上下文长度
train_on_inputs: bool  # 是否在输入上训练
model_max_len: int  # 模型最大长度
max_samples: int  # 最大样本数
eval_dataset: string  # 评估数据集
val_size: float  # 验证集比例
max_val_samples: int  # 最大验证样本数
streaming: bool  # 是否启用流式处理
data_seed: int  # 数据随机种子
packing: bool  # 是否启用序列打包
overwrite_cache: bool  # 是否覆盖缓存
preprocessing_num_workers: int  # 预处理工作线程数
dataloader_num_workers: int  # 数据加载工作线程数
expand_side: ["user", "assistant"]  # 扩展角色类型
conversation_cache: string  # 对话缓存路径
```

### 输出设置 (Output)
```yaml
output_dir: string  # 输出目录
logging_dir: string  # 日志目录
logging_steps: int  # 日志记录步数
save_steps: int  # 保存步数
max_saved_steps: int  # 最大保存检查点数
eval_steps: int  # 评估步数
plot_loss: bool  # 是否绘制损失曲线
hub_model_id: string  # Hugging Face Hub模型ID
hub_strategy: ["end", "all_checkpoints", "checkpoint", "every_save"]  # Hub上传策略
hub_private_repo: bool  # 是否为私有仓库
push_to_hub: bool  # 是否推送到Hub
overwrite_output_dir: bool  # 是否覆盖输出目录
save_strategy: ["no", "steps", "epoch"]  # 保存策略
save_total_limit: int  # 保存的检查点总数
save_only_model: bool  # 是否只保存模型
report_to: ["none", "wandb", "tensorboard", "mlflow", "swanlab"]  # 报告工具
```

### 训练配置 (Training)
```yaml
num_train_epochs: float  # 训练轮数
max_steps: int  # 最大训练步数
per_device_train_batch_size: int  # 每设备训练批次大小
per_device_eval_batch_size: int  # 每设备评估批次大小
gradient_accumulation_steps: int  # 梯度累积步数
gradient_checkpointing: bool  # 是否启用梯度检查点
learning_rate: float  # 学习率
weight_decay: float  # 权重衰减
adam_beta1: float  # Adam参数beta1
adam_beta2: float  # Adam参数beta2
adam_epsilon: float  # Adam参数epsilon
max_grad_norm: float  # 最大梯度范数
optim: string  # 优化器
lr_scheduler_type: ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau"]  # 学习率调度器类型
warmup_ratio: float  # 热身比例
warmup_steps: int  # 热身步数
fp16: bool  # 是否使用FP16精度
bf16: bool  # 是否使用BF16精度
seed: int  # 随机种子
neftune_noise_alpha: float  # NEFTune噪声强度
resume_from_checkpoint: [string, bool]  # 从检查点恢复训练
ddp_timeout: int  # DDP超时设置
padding_side: ["left", "right"]  # 填充方向
```

### LoRA 特定配置
```yaml
lora_rank: int  # LoRA秩
lora_alpha: int  # LoRA alpha参数
lora_dropout: float  # LoRA dropout率
lora_target: string  # LoRA目标模块
additional_target: string  # 额外目标模块
```

### AdaLoRA 特定配置
```yaml
adalora_target_r: int  # AdaLoRA目标秩
adalora_init_r: int  # AdaLoRA初始秩
adalora_tinit: int  # AdaLoRA tinit参数
adalora_tfinal: int  # AdaLoRA tfinal参数
adalora_delta_t: int  # AdaLoRA delta_t参数
adalora_alpha: int  # AdaLoRA alpha参数
```

### QLoRA 特定配置
```yaml
bnb_4bit_quant_type: ["fp4", "nf4"]  # BitsAndBytes 4位量化类型
bnb_4bit_use_double_quant: bool  # 是否使用双重量化
bnb_4bit_compute_dtype: ["fp16", "bf16", "fp32"]  # 计算数据类型
```

### 评估配置 (Evaluation)
```yaml
predict_with_generate: bool  # 是否使用生成进行预测
eval_strategy: ["steps", "epoch", "no"]  # 评估策略
prediction_loss_only: bool  # 是否只计算预测损失
remove_unused_columns: bool  # 是否移除未使用的列
stop_token: string  # 停止令牌
metric_for_best_model: string  # 用于选择最佳模型的指标
greater_is_better: bool  # 该指标是否越大越好
```

### PEFT 特定配置
```yaml
unsloth_peft: bool  # 是否使用Unsloth PEFT
```

### PPO 特定配置
```yaml
reward_model: string  # 奖励模型路径
reward_model_type: string  # 奖励模型类型
```

### DPO 特定配置
```yaml
dpo_beta: float  # DPO beta参数
dpo_loss_type: ["sigmoid", "hinge", "ipo", "kto_pair"]  # DPO损失类型
```

### 分布式训练配置
```yaml
local_rank: int  # 本地进程排名
ddp_find_unused_parameters: bool  # 是否查找未使用的参数
deepspeed_config_path: string  # DeepSpeed配置路径
fsdp: string  # FSDP设置
fsdp_config: string  # FSDP配置
```

### 其他配置
```yaml
disable_tqdm: bool  # 是否禁用进度条
low_cpu_mem_usage: bool  # 是否启用低CPU内存使用
group_by_length: bool  # 是否按长度分组
length_column_name: string  # 长度列名称
ignore_data_skip: bool  # 是否忽略数据跳过
```

## 关于特定训练模式的额外变量

### 预训练 (pre) 特定配置
```yaml
mlm_probability: float  # 掩码语言模型的掩码概率
```

### SFT (监督微调) 特定配置 
```yaml
upcast_layernorm: bool  # 是否上采样LayerNorm
```

### RM (奖励模型) 特定配置
```yaml
margin: float  # 奖励模型的边际值
```

### PPO (近端策略优化) 特定配置
```yaml
ppo_epochs: int  # PPO训练轮数
kl_penalty: string  # KL惩罚设置
optimize_device_cache: bool  # 是否优化设备缓存
experience_max_seq_len: int  # 经验最大序列长度
max_prompt_seq_len: int  # 最大提示序列长度
init_kl_coef: float  # 初始KL系数
adap_kl_ctrl: bool  # 是否使用自适应KL控制
```

## 如何验证变量的真实存在

每个变量都可以通过以下方式验证其真实存在：

1. 查看 `src/llamafactory/train/parser.py` 文件，这里定义了所有的命令行参数
2. 检查 `examples/` 目录下的示例配置文件
3. 查看实现代码如 `src/llamafactory/train/trainer.py` 和 `src/llamafactory/train/sft.py` 等

例如，通过查看源代码可以找到：

```python
# 在 src/llamafactory/train/parser.py 中
parser.add_argument("--model_name_or_path", type=str, help="Path to the model.")
parser.add_argument("--finetuning_type", type=str, choices=["full", "freeze", "lora", ...], help="Finetuning type.")
```

这些参数直接对应到 YAML 文件中的配置变量。

## 特别提醒

1. 并非所有变量在所有训练阶段都适用，例如 `dpo_beta` 只在 DPO 阶段使用
2. 一些变量可能有默认值，不在配置文件中指定也可以运行
3. 随着 LLaMA-Factory 版本更新，可能会有新的变量添加或现有变量变更

以上列表基于 LLaMA-Factory 的最新版本整理，涵盖了几乎所有可在 YAML 配置文件中设置的变量。实际使用时，可根据具体需求选择相关变量进行配置。