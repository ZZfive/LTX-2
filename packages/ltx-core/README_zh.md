# LTX-Core

LTX-2 音视频生成模型的核心基础库。本包包含原始模型定义、组件实现以及 `ltx-pipelines` 和 `ltx-trainer` 所使用的权重加载逻辑。

## 📦 模块结构

- **`components/`**：遵循标准协议的模块化扩散组件（调度器、引导器、加噪器、Patchifier）
- **`conditioning/`**：用于准备潜在状态并应用条件控制（图像、视频、关键帧）的工具
- **`guidance/`**：用于对注意力机制进行细粒度控制的扰动系统
- **`loader/`**：从 `.safetensors` 加载权重、融合 LoRA 以及管理内存的工具
- **`model/`**：LTX-2 Transformer、Video VAE、Audio VAE、声码器（Vocoder）和超分辨率模型的 PyTorch 实现
- **`text_encoders/gemma`**：Gemma 文本编码器实现，包含分词器、特征提取器，以及用于音视频联合生成和纯视频生成的独立编码器
- **`quantization/`**：FP8 量化后端（FP8-TensorRT-LLM scaled MM、FP8 cast），用于降低内存占用

## 🚀 快速入门

`ltx-core` 提供构建推理流程所需的基本模块（模型、组件和工具）。如需开箱即用的推理 Pipeline，请使用 [`ltx-pipelines`](../ltx-pipelines/)；如需训练，请使用 [`ltx-trainer`](../ltx-trainer/)。

## 🔧 安装

```bash
# 从仓库根目录安装
uv sync --frozen

# 或作为独立包安装
pip install -e packages/ltx-core
```

---

## 构建模块概览

`ltx-core` 提供可自由组合的模块化组件，用于构建自定义推理流程。

### 核心模型

- **Transformer** ([`model/transformer/`](src/ltx_core/model/transformer/))：LTX-2 非对称双流 Transformer（视频流 14B 参数，音频流 5B 参数），具有双向跨模态注意力机制，实现音视频联合处理。输入格式为 [`Modality`](src/ltx_core/model/transformer/modality.py)
- **Video VAE** ([`model/video_vae/`](src/ltx_core/model/video_vae/))：将视频像素编码/解码至潜在空间，支持时间和空间压缩
- **Audio VAE** ([`model/audio_vae/`](src/ltx_core/model/audio_vae/))：将音频声谱图编码/解码至潜在空间
- **声码器（Vocoder）** ([`model/audio_vae/`](src/ltx_core/model/audio_vae/))：将梅尔声谱图转换为音频波形的神经声码器
- **文本编码器** ([`text_encoders/`](src/ltx_core/text_encoders/))：基于 Gemma 3 的多语言编码器，支持多层特征提取和 Thinking Token，为视频和音频条件控制分别生成独立的嵌入表示
- **空间超分辨率模块（Spatial Upscaler）** ([`model/upsampler/`](src/ltx_core/model/upsampler/))：对潜在表示进行上采样，支持更高分辨率的生成

### 扩散组件

- **调度器（Schedulers）** ([`components/schedulers.py`](src/ltx_core/components/schedulers.py))：噪声调度策略（LTX2Scheduler、LinearQuadratic、Beta），控制去噪过程
- **引导器（Guiders）** ([`components/guiders.py`](src/ltx_core/components/guiders.py))：引导策略（CFG、STG、APG），控制生成质量及提示词遵循度
- **加噪器（Noisers）** ([`components/noisers.py`](src/ltx_core/components/noisers.py))：按照扩散调度向潜在变量添加噪声
- **Patchifier** ([`components/patchifiers.py`](src/ltx_core/components/patchifiers.py))：在空间潜在格式 `[B, C, F, H, W]` 与 Transformer 所需的序列格式 `[B, seq_len, dim]` 之间进行转换

### 条件控制

- **Conditioning** ([`conditioning/`](src/ltx_core/conditioning/))：用于准备和应用各类条件控制（图像、视频、关键帧）的工具
- **Guidance** ([`guidance/`](src/ltx_core/guidance/))：注意力机制细粒度扰动系统（如跳过特定注意力层）

### 工具模块

- **Loader** ([`loader/`](src/ltx_core/loader/))：从 `.safetensors` 加载模型、融合 LoRA、权重重映射及内存管理
- **Quantization** ([`quantization/`](src/ltx_core/quantization/))：FP8 量化后端，降低内存占用，加速推理

---

### 模型加载器（Loader）

`loader/` 模块提供 `SingleGPUModelBuilder`，这是一个冻结的数据类，负责从 `.safetensors` 检查点加载 PyTorch 模型，并可选地融合一个或多个 LoRA 适配器。

#### 基本用法

```python
from ltx_core.loader import SingleGPUModelBuilder

builder = SingleGPUModelBuilder(
    model_class_configurator=MyModelConfigurator,
    model_path="/path/to/model.safetensors",
)
model = builder.build(device=torch.device("cuda"))
```

#### 加载 LoRA 适配器

在调用 `.build()` 之前，使用 `.lora()` 方法挂载一个或多个 LoRA 适配器：

```python
builder = (
    SingleGPUModelBuilder(
        model_class_configurator=MyModelConfigurator,
        model_path="/path/to/model.safetensors",
    )
    .lora("/path/to/lora_a.safetensors", strength=0.8)
    .lora("/path/to/lora_b.safetensors", strength=0.5)
)
model = builder.build(device=torch.device("cuda"))
```

#### 内存高效 LoRA 加载（`lora_load_device`）

默认情况下，LoRA 权重加载到 **CPU**（`lora_load_device=torch.device("cpu")`）。这意味着每个 LoRA 适配器保存在 CPU 内存中，在权重融合期间依次传输到 GPU，即使融合大型适配器也能保持较低的 GPU 峰值内存。

如果所有适配器可以同时放入 GPU 内存，可以将 `lora_load_device` 设置为目标 CUDA 设备，跳过 CPU 暂存阶段：

```python
import torch
from ltx_core.loader import SingleGPUModelBuilder

# 直接将 LoRA 权重加载到 GPU（更快，但占用更多 GPU 内存）
builder = SingleGPUModelBuilder(
    model_class_configurator=MyModelConfigurator,
    model_path="/path/to/model.safetensors",
    lora_load_device=torch.device("cuda"),
).lora("/path/to/lora.safetensors", strength=1.0)

model = builder.build(device=torch.device("cuda"))
```

---

### 量化（Quantization）

`quantization/` 模块为 LTX-2 Transformer 提供 FP8 量化支持，在保持生成质量的同时显著降低内存占用。提供两种后端：

#### FP8 Scaled MM（TensorRT-LLM）

使用 NVIDIA TensorRT-LLM 的 `cublas_scaled_mm` 实现高效的 FP8 矩阵乘法。权重以 FP8 格式存储并附带逐张量缩放因子，输入支持动态量化（无需校准数据）或静态量化（使用校准文件）。

**依赖安装**：`uv sync --frozen --extra fp8-trtllm`

**通过 QuantizationPolicy 使用**：

```python
from ltx_core.quantization import QuantizationPolicy

# 动态输入量化（无需校准）
policy = QuantizationPolicy.fp8_scaled_mm()

# 静态输入量化（需提供校准文件）
policy = QuantizationPolicy.fp8_scaled_mm(calibration_amax_path="/path/to/amax.json")
```

将 `sd_ops` 和 `module_ops` 传递给模型构建器：

```python
from ltx_core.loader import SingleGPUModelBuilder

builder = SingleGPUModelBuilder(
    model=model,
    device=device,
    sd_ops=policy.sd_ops,
    module_ops=policy.module_ops,
)
builder.load(checkpoint_path)
```

**校准文件格式**（用于静态输入量化）：

```json
{
  "amax_values": {
    "transformer_blocks.0.attn.to_q.input_quantizer": 12.5,
    "transformer_blocks.0.attn.to_k.input_quantizer": 8.3,
    ...
  }
}
```

#### FP8 Cast

一种更简单的方案：将权重转换为 FP8 格式存储，推理时再反向转换：

```python
policy = QuantizationPolicy.fp8_cast()
```

如需将上述模块组合为完整、生产级的 Pipeline 实现，请参阅 [`ltx-pipelines`](../ltx-pipelines/) 包。

---

# 架构详解

本节对 LTX-2 音视频生成模型的内部架构进行深入介绍。

## 目录

1. [高层架构](#高层架构)
2. [Transformer](#transformer)
3. [Video VAE](#video-vae)
4. [Audio VAE](#audio-vae)
5. [文本编码（Gemma）](#文本编码gemma)
6. [空间超分辨率模块](#空间超分辨率模块)
7. [数据流](#数据流)

---

## 高层架构

LTX-2 是一个**非对称双流扩散 Transformer**，对视频和音频信号进行文本条件下的联合建模，捕捉真实的跨模态联合依赖关系（区别于顺序执行的 T2V → V2A Pipeline）。

### 核心设计原则

- **解耦潜在表示**：模态专属的独立 VAE，支持视频侧 3D RoPE 与音频侧 1D RoPE 的独立优化，以及原生的 V2A/A2V 编辑工作流
- **非对称双流**：视频流（14B 参数，负责时空动态）+ 音频流（5B 参数，负责 1D 时序），共享 48 个 Transformer Block，但隐层宽度不同
- **双向跨模态注意力**：1D 时序 RoPE 实现亚帧级对齐，将视觉线索映射到听觉事件（唇形同步、音效、环境声学）
- **跨模态 AdaLN**：缩放/偏移参数由另一模态的隐藏状态条件化，实现不同扩散时间步/时序分辨率下的跨模态同步

```text
┌─────────────────────────────────────────────────────────────┐
│                       输入准备阶段                           │
│                                                             │
│  视频像素 → Video VAE 编码器 → 视频潜在变量                   │
│  音频波形 → Audio VAE 编码器 → 音频潜在变量                   │
│  文本提示 → Gemma 3 编码器 → 文本嵌入                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│      LTX-2 非对称双流 Transformer（48 个 Block）              │
│                                                             │
│  ┌──────────────────────┐      ┌──────────────────────┐     │
│  │   视频流（14B）        │      │   音频流（5B）         │     │
│  │                      │      │                      │     │
│  │  3D RoPE (x,y,t)     │      │  1D RoPE（时序）       │     │
│  │                      │      │                      │     │
│  │  自注意力             │      │  自注意力              │     │
│  │  文本交叉注意力        │      │  文本交叉注意力         │     │
│  │                      │◄────►│                      │     │
│  │  音视频交叉注意力       │      │  音视频交叉注意力        │     │
│  │  （1D 时序 RoPE）      │      │  （1D 时序 RoPE）      │     │
│  │  跨模态 AdaLN         │      │  跨模态 AdaLN          │     │
│  │  前馈网络（FFN）        │      │  前馈网络（FFN）         │     │
│  └──────────────────────┘      └──────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       输出解码阶段                           │
│                                                             │
│  视频潜在变量 → Video VAE 解码器 → 视频像素                   │
│  音频潜在变量 → Audio VAE 解码器 → 梅尔声谱图                  │
│  梅尔声谱图 → 声码器 → 音频波形（24 kHz）                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Transformer

LTX-2 的核心是一个包含 **48 层的非对称双流扩散 Transformer**，可同时处理视频和音频 Token。该架构将 14B 参数分配给视频流，5B 参数分配给音频流，反映了两种模态不同的信息密度。

### 模型结构

**源码**：[`src/ltx_core/model/transformer/model.py`](src/ltx_core/model/transformer/model.py)

`LTXModel` 类实现了该 Transformer，同时支持纯视频和音视频联合生成模式。实际使用时，请参阅 [`ltx-pipelines`](../ltx-pipelines/) 包，其中封装了模型加载和初始化逻辑。

### Transformer Block 结构

**源码**：[`src/ltx_core/model/transformer/transformer.py`](src/ltx_core/model/transformer/transformer.py)

每个双流 Block 依次执行四个操作：

1. **自注意力（Self-Attention）**：各流内部的模态内注意力
2. **文本交叉注意力（Text Cross-Attention）**：对两个流施加文本提示条件
3. **音视频交叉注意力（Audio-Visual Cross-Attention）**：双向跨模态信息交换
4. **前馈网络（FFN）**：特征精炼

```text
┌─────────────────────────────────────────────────────────────┐
│                     TRANSFORMER BLOCK                       │
│                                                             │
│  视频流（14B）: 输入 → RMSNorm → AdaLN → 自注意力 →           │
│               RMSNorm → 文本交叉注意力 →                     │
│               RMSNorm → AdaLN → 音视频交叉注意力（1D RoPE）→  │
│               RMSNorm → AdaLN → FFN → 输出                  │
│                                                             │
│  音频流（5B）: 输入 → RMSNorm → AdaLN → 自注意力 →            │
│               RMSNorm → 文本交叉注意力 →                     │
│               RMSNorm → AdaLN → 音视频交叉注意力（1D RoPE）→  │
│               RMSNorm → AdaLN → FFN → 输出                  │
│                                                             │
│  RoPE：视频=3D (x,y,t)，音频=1D (t)，交叉注意力=1D (t)       │
│  AdaLN：由时间步条件化，音视频交叉注意力部分为跨模态条件化        │
└─────────────────────────────────────────────────────────────┘
```

### 音视频交叉注意力详解

双向交叉注意力实现了紧密的时序对齐：视频流和音频流通过 1D 时序 RoPE 进行双向信息交换（仅时序同步，无空间对齐）。AdaLN 门控以各自模态的时间步为条件，实现跨模态同步。

### 扰动机制（Perturbations）

Transformer 支持[**扰动**](src/ltx_core/guidance/perturbations.py)，可选择性地跳过某些注意力操作。

扰动机制允许在推理时禁用特定的注意力机制，对于 STG（时空引导）等引导技术非常有用。

**支持的扰动类型**：

- `SKIP_VIDEO_SELF_ATTN`：跳过视频自注意力
- `SKIP_AUDIO_SELF_ATTN`：跳过音频自注意力
- `SKIP_A2V_CROSS_ATTN`：跳过音频到视频的交叉注意力
- `SKIP_V2A_CROSS_ATTN`：跳过视频到音频的交叉注意力

扰动机制由 STG（时空引导）等引导机制在内部使用。使用示例请参阅 [`ltx-pipelines`](../ltx-pipelines/) 包。

---

## Video VAE

Video VAE（[`src/ltx_core/model/video_vae/`](src/ltx_core/model/video_vae/)）负责将视频像素编码为潜在表示，并将其解码还原。

### 架构

- **编码器**：将 `[B, 3, F, H, W]` 像素压缩为 `[B, 128, F', H/32, W/32]` 潜在变量
  - 其中 `F' = 1 + (F-1)/8`（帧数须满足 `(F-1) % 8 == 0`）
  - 示例：`[B, 3, 33, 512, 512]` → `[B, 128, 5, 16, 16]`
- **解码器**：将 `[B, 128, F, H, W]` 潜在变量还原为 `[B, 3, F', H*32, W*32]` 像素
  - 其中 `F' = 1 + (F-1)*8`
  - 示例：`[B, 128, 5, 16, 16]` → `[B, 3, 33, 512, 512]`

Video VAE 在 Pipeline 内部用于将视频像素编码为潜在变量，以及将潜在变量解码回像素。使用示例请参阅 [`ltx-pipelines`](../ltx-pipelines/) 包。

---

## Audio VAE

Audio VAE（[`src/ltx_core/model/audio_vae/`](src/ltx_core/model/audio_vae/)）负责处理音频声谱图。

### 架构

紧凑的神经音频表示，针对扩散训练进行了优化，原生支持立体声：在编码前对双通道梅尔声谱图（16 kHz 输入）进行通道拼接。

- **编码器**：`[B, mel_bins, T]` → `[B, 8, T/4, 16]` 潜在变量（4× 时序下采样，8 通道，潜在空间中 16 个 mel bin，每个 Token 约 1/25 秒，128 维特征向量）
- **解码器**：`[B, 8, T, 16]` → `[B, mel_bins, T*4]` 梅尔声谱图
- **声码器（Vocoder）**：基于 HiFi-GAN，针对立体声合成和上采样进行改进（16 kHz 梅尔 → 24 kHz 波形，为支持立体声将生成器容量翻倍）

**下采样倍率**：
- 时序：4×（时间步）
- 频率：可变（输入 mel_bins → 潜在空间中固定为 16）

Audio VAE 在 Pipeline 内部用于将梅尔声谱图编码为潜在变量，以及将潜在变量解码回梅尔声谱图；声码器负责将梅尔声谱图转换为音频波形。使用示例请参阅 [`ltx-pipelines`](../ltx-pipelines/) 包。

---

## 文本编码（Gemma）

LTX-2 使用 **Gemma 3**（Gemma 3-12B）作为多语言文本编码器主干，位于 [`src/ltx_core/text_encoders/gemma/`](src/ltx_core/text_encoders/gemma/)。先进的文本理解能力不仅对全球语言支持至关重要，也对生成语音的语音和语义准确性起到关键作用。

### 文本编码器架构

文本条件控制 Pipeline 由三个阶段组成：

1. **Gemma 3 主干**：仅解码器架构的 LLM，处理文本 Token → 跨所有层的嵌入 `[B, T, D, L]`
2. **多层特征提取器**：聚合来自所有解码器层（而非仅最后一层）的特征，应用均值中心化缩放，展平为 `[B, T, D×L]`，并通过可学习矩阵 W 进行投影（与 LTX-2 联合优化，LLM 权重保持冻结）
3. **文本连接器（Text Connector）**：带有可学习 Register（取代填充位置，论文中也称为"Thinking Token"）的双向 Transformer Block，用于上下文混合。视频流和音频流使用独立的连接器（`Embeddings1DConnector`）

**编码器类型**：

- `AVGemmaTextEncoderModel`：音视频联合生成（两个连接器 → `AVGemmaEncoderOutput`，包含独立的视频/音频上下文）
- `VideoGemmaTextEncoderModel`：纯视频生成（单个连接器 → `VideoGemmaEncoderOutput`）

### 系统提示词

系统提示词用于增强用户输入的提示词：

- **文本到视频（T2V）**：[`gemma_t2v_system_prompt.txt`](src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_t2v_system_prompt.txt)
- **图像到视频（I2V）**：[`gemma_i2v_system_prompt.txt`](src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_i2v_system_prompt.txt)

> **重要提示**：即使来自相同的提示词，视频和音频接收到的上下文嵌入也是**不同的**。这使得模型能够针对各模态进行更专属的条件控制，并在合成自然的音节节奏、口音和情感语调的同时，实现与视觉唇形运动的精准同步。

**输出格式**：

- 视频上下文：`[B, seq_len, 4096]`——视频专属文本嵌入
- 音频上下文：`[B, seq_len, 2048]`——音频专属文本嵌入

文本编码器在 Pipeline 内部使用。使用示例请参阅 [`ltx-pipelines`](../ltx-pipelines/) 包。

---

## 空间超分辨率模块

超分辨率模块（[`src/ltx_core/model/upsampler/`](src/ltx_core/model/upsampler/)）对潜在表示进行上采样，输出更高分辨率的结果。

空间超分辨率模块在两阶段 Pipeline 内部使用（如 [`TI2VidTwoStagesPipeline`](../ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py)、[`ICLoraPipeline`](../ltx-pipelines/src/ltx_pipelines/ic_lora.py)），在最终 VAE 解码前对低分辨率潜在变量进行上采样。使用示例请参阅 [`ltx-pipelines`](../ltx-pipelines/) 包。

---

## 数据流

### 完整生成 Pipeline

以下是所有组件协同工作的概念流程（[`src/ltx_core/components/`](src/ltx_core/components/)）：

**Pipeline 步骤**：

1. **文本编码**：文本提示 → Gemma 编码器 → 独立的视频/音频嵌入
2. **潜在变量初始化**：以空间格式 `[B, C, F, H, W]` 初始化噪声潜在变量
3. **Patchification**：将空间潜在变量转换为序列格式 `[B, seq_len, dim]`，供 Transformer 处理
4. **Sigma 调度**：生成噪声调度（自适应 Token 数量）
5. **去噪循环**：使用 Transformer 预测迭代去噪
   - 构造带有逐 Token 时间步和 RoPE 位置的 Modality 输入
   - 通过 Transformer 进行前向推理（CFG 需要条件和无条件两路）
   - 应用引导策略（CFG、STG 等）
   - 使用扩散步骤（Euler 等）更新潜在变量
6. **Unpatchification**：将序列格式转换回空间格式
7. **VAE 解码**：将潜在变量解码为像素空间（两阶段 Pipeline 可选上采样）

**可用 Pipeline**：

- [`TI2VidTwoStagesPipeline`](../ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py)——两阶段文本到视频（推荐）
- [`ICLoraPipeline`](../ltx-pipelines/src/ltx_pipelines/ic_lora.py)——使用 IC-LoRA 控制的视频到视频
- [`DistilledPipeline`](../ltx-pipelines/src/ltx_pipelines/distilled.py)——使用蒸馏模型的快速推理
- [`KeyframeInterpolationPipeline`](../ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py)——基于关键帧的插帧

详细使用示例请参阅 [ltx-pipelines README](../ltx-pipelines/README.md)。

---

## 🔗 相关项目

- **[ltx-pipelines](../ltx-pipelines/)**——文本到视频、图像到视频、视频到视频的高层 Pipeline 实现
- **[ltx-trainer](../ltx-trainer/)**——训练与微调工具
