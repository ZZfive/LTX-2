from dataclasses import dataclass, replace
from typing import NamedTuple

import torch


class VideoPixelShape(NamedTuple):  # 表示视频像素空间尺寸
    """
    Shape of the tensor representing the video pixel array. Assumes BGR channel format.
    """

    batch: int
    frames: int
    height: int
    width: int
    fps: float


class SpatioTemporalScaleFactors(NamedTuple):  # 表示时空缩放因子，记录像素视频转化为VAE latents在时间、高、宽三个维度上的下采样率
    """
    Describes the spatiotemporal downscaling between decoded video space and
    the corresponding VAE latent grid.
    Field order matches the (frame/time, height, width) axis layout used by
    latent tensors and meshgrid coordinates elsewhere in the codebase.
    """

    time: int
    height: int
    width: int

    @classmethod
    def default(cls) -> "SpatioTemporalScaleFactors":
        return cls(time=8, height=32, width=32)  # 默认值

    @classmethod
    def from_blocks(cls, blocks: list, patch_size: int) -> "SpatioTemporalScaleFactors":  # 从VAE encoder/decoder block list中推导出时空缩放因子
        """Derive the scale factors from a VAE encoder/decoder block list.
        Each ``compress_*`` block halves (encoder) or doubles (decoder) its target
        axes by a stride of 2, independent of any channel ``multiplier``. The initial
        patchify contributes an extra ``patch_size`` of spatial compression. Deriving
        the factors from the blocks keeps a single source of truth that stays correct
        across VAE variants (e.g. the 32x32x8 default and the 16x16x4 variant) instead
        of relying on a hardcoded constant.

        从 VAE 编码器/解码器的 block 列表推导时空缩放因子。
        每个 ``compress_*`` block（通常 encoder 用来下采样，decoder 用来上采样）都会以 stride=2 对目标轴做二分（encoder）或二倍（decoder），
        与通道 ``multiplier`` 无关。最初的 patchify 操作还会带来一个 ``patch_size`` 的空间压缩。
        从这些 blocks 中推导缩放因子，可以作为单一的真实来源，使不同的 VAE 变体（比如 32x32x8 默认和 16x16x4 变体）都能正确得到对应的值，
        避免依赖硬编码常数。
        """
        spatial_steps = 0
        temporal_steps = 0
        for block_name, _ in blocks:
            if block_name.startswith(("compress_space", "compress_all")): # 空间下采样
                spatial_steps += 1
            if block_name.startswith(("compress_time", "compress_all")): # 时间下采样
                temporal_steps += 1
        spatial = patch_size * (2**spatial_steps)
        return cls(time=2**temporal_steps, height=spatial, width=spatial)  # 返回时空缩放因子

    @classmethod
    def from_model_config(cls, model_config: dict) -> "SpatioTemporalScaleFactors":
        """Derive the video scale factors from a checkpoint's model config dict.
        Reads the embedded VAE block list (see ``from_blocks``). Falls back to the
        default when the config carries no VAE block list -- either no ``vae`` section
        or a ``vae`` section without encoder/decoder blocks (e.g. audio-only
        checkpoints), where video tools are never built.
        """
        vae_config = model_config.get("vae", {})
        blocks = vae_config.get("encoder_blocks") or vae_config.get("decoder_blocks")
        if not blocks:
            return cls.default()
        return cls.from_blocks(blocks, vae_config.get("patch_size", 4))  # 从模型配置中推导时空缩放因子


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class VideoLatentShape(NamedTuple):  # 表示视频隐空间尺寸，记录VAE隐空间中视频的batch、通道、帧数、高、宽
    """
    Shape of the tensor representing video in VAE latent space.
    The latent representation is a 5D tensor with dimensions ordered as
    (batch, channels, frames, height, width). Spatial and temporal dimensions
    are downscaled relative to pixel space according to the VAE's scale factors.

    表示视频在VAE隐空间中的张量形状。
    该隐空间表示为一个5维张量，维度顺序为 (batch, channels, frames, height, width)。
    与像素空间相比，空间和时间维度会按VAE的缩放因子进行下采样。
    """

    batch: int
    channels: int
    frames: int
    height: int
    width: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.height, self.width])

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "VideoLatentShape":
        return VideoLatentShape(
            batch=shape[0],
            channels=shape[1],
            frames=shape[2],
            height=shape[3],
            width=shape[4],
        )

    def token_count(self) -> int:
        """Number of tokens after patchification with the default patch size of 1."""
        return self.frames * self.height * self.width  # 返回视频隐空间中的token数量，即帧数*高*宽

    def mask_shape(self) -> "VideoLatentShape":
        return self._replace(channels=1)  # 将通道数替换为1，表示只保留一个通道，通常用于掩码操作

    @staticmethod
    def from_pixel_shape(  # 从像素空间形状转换为视频隐空间形状
        shape: VideoPixelShape,
        latent_channels: int = 128,  # 默认通道数为128
        scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
    ) -> "VideoLatentShape":
        frames = (shape.frames - 1) // scale_factors.time + 1  # 计算视频隐空间中的帧数，即像素空间帧数除以时间缩放因子并向上取整
        height = shape.height // scale_factors.height  # 计算视频隐空间中的高度，即像素空间高度除以空间缩放因子
        width = shape.width // scale_factors.width  # 计算视频隐空间中的宽度，即像素空间宽度除以空间缩放因子

        return VideoLatentShape(
            batch=shape.batch,
            channels=latent_channels,
            frames=frames,
            height=height,
            width=width,
        )

    def upscale(self, scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS) -> "VideoLatentShape":
        return self._replace(
            channels=3,
            frames=(self.frames - 1) * scale_factors.time + 1,
            height=self.height * scale_factors.height,
            width=self.width * scale_factors.width,
        )  # 从隐空间映射回像素空间


class AudioLatentShape(NamedTuple):  # 表示音频隐空间尺寸
    """
    Shape of audio in VAE latent space: (batch, channels, frames, mel_bins).
    mel_bins is the number of frequency bins from the mel-spectrogram encoding.

    表示音频在VAE隐空间中的张量形状。
    该隐空间表示为一个4维张量，维度顺序为 (batch, channels, frames, mel_bins)。
    mel_bins 是梅尔频谱图编码的频率 bin 数量。
    """

    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.mel_bins])

    def token_count(self) -> int:
        """Number of tokens after patchification."""  # 返回音频隐空间中的token数量，即帧数
        return self.frames

    def mask_shape(self) -> "AudioLatentShape":
        return self._replace(channels=1, mel_bins=1)  # 将通道数和梅尔频谱图bin数替换为1，表示只保留一个通道和梅尔频谱图bin

    @staticmethod
    def from_torch_shape(shape: torch.Size) -> "AudioLatentShape":
        return AudioLatentShape(
            batch=shape[0],  # 批量大小
            channels=shape[1],  # 通道数
            frames=shape[2],  # 帧数
            mel_bins=shape[3],  # 梅尔频谱图bin数
        )

    @staticmethod
    def from_duration(
        batch: int,  # 批量大小
        duration: float,  # 音频时长（秒）
        channels: int = 8,  # 通道数
        mel_bins: int = 16,  # 梅尔频谱图bin数
        sample_rate: int = 16000,  # 音频采样率（Hz），此处每秒 16000 个采样点
        hop_length: int = 160,  # STFT 的帧移，即每帧之间间隔采样点数量，此处间隔为 160 个采样点
        audio_latent_downsample_factor: int = 4,  # 音频 VAE 在时间轴上的下采样倍率，此处为4倍
    ) -> "AudioLatentShape":
        latents_per_second = float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)  # 计算每秒音频帧数，即采样率除以帧移除以下采样倍率

        return AudioLatentShape(
            batch=batch,  # 批量大小
            channels=channels,  # 通道数
            frames=round(duration * latents_per_second),  # 计算音频帧数，此处 duration 为音频时长（秒），latents_per_second 为每秒音频帧数
            mel_bins=mel_bins,  # 梅尔频谱图bin数
        )

    @staticmethod
    def from_video_pixel_shape(
        shape: VideoPixelShape,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> "AudioLatentShape":
        return AudioLatentShape.from_duration(
            batch=shape.batch,
            duration=float(shape.frames) / float(shape.fps),  # 计算音频时长，即帧数除以帧率
            channels=channels,
            mel_bins=mel_bins,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio_latent_downsample_factor=audio_latent_downsample_factor,
        )


@dataclass(frozen=True)
class Audio:
    """
    Container for decoded audio samples and metadata.
    Attributes:
        waveform: Audio waveform tensor.
        sampling_rate: Sampling rate (Hz) of the waveform.
    """

    waveform: torch.Tensor  # 音频波形张量
    sampling_rate: int  # 采样率（Hz）

    def to(self, **kwargs: object) -> "Audio":
        return replace(self, waveform=self.waveform.to(**kwargs))  # 将音频波形张量转换为指定设备


@dataclass(frozen=True)
class GeneratedKeyframeLayout:  # 生成关键帧槽在 token 序列中的定位书签
    """Index of generated-keyframe slot tokens inside a :class:`LatentState` sequence.

    Written once by :class:`~ltx_core.conditioning.types.keyframe_slots.VideoGeneratedKeyframeSlots`
    at the moment it appends the slots. Later conditionings may append more tokens after those
    slots, so the slots are **not** guaranteed to be the trailing slice -- this layout is the
    only exact way to find them again.

    Typical sequence after slots + later conditionings::

        [target-grid tokens] [slot0] [slot1] ... [other cond tokens]
                             ^ first_token
                             <---- token_slice ---->

    Read back by :meth:`ltx_core.tools.VideoLatentTools.extract_generated_keyframes` /
    ``clear_conditioning``: slice ``latent[:, token_slice]``, split every
    ``tokens_per_keyframe`` tokens, unpatchify each chunk as a 1-frame latent.

    Attributes:
        pixel_frame_indices: Pixel-frame index each slot should generate, in token order
            (strictly increasing). These are pixel-space times, not latent-frame indices.
        tokens_per_keyframe: Tokens occupied by one slot (= one latent frame at the target
            spatial resolution).
        first_token: Index of the first slot token, recorded as the sequence length *before*
            the slots were appended.

    生成关键帧槽在 ``LatentState`` token 序列中的定位书签，不是槽内容本身。

    由 ``VideoGeneratedKeyframeSlots`` 在追加槽的当下写入。之后其它条件项仍会往序列尾部
    append，所以槽**不一定是最后一段 token**，不能靠「取尾巴」找回，只能靠这份 layout。

    槽追加后再跟其它条件时的典型布局::

        [target 网格 token] [slot0] [slot1] ... [其它条件 token]
                            ^ first_token
                            <---- token_slice ---->

    读出点是 ``VideoLatentTools.extract_generated_keyframes`` / ``clear_conditioning``：
    用 ``token_slice`` 切开 ``latent``，按 ``tokens_per_keyframe`` 拆成 K 段，各自
    unpatchify 成单帧 latent。

    属性：
        pixel_frame_indices: 每个槽要生成的**像素帧**下标（按 token 顺序，严格递增），不是 latent 帧号。
        tokens_per_keyframe: 一个槽占多少 token（= target 空间分辨率下 1 个 latent frame 的 token 数）。
        first_token: 第一个槽 token 的下标，取的是**追加槽之前**的序列长度。
    """

    pixel_frame_indices: tuple[int, ...]  # 各槽对应的像素帧下标，token 顺序，严格递增
    tokens_per_keyframe: int  # 单个槽的 token 数 = 1 个 latent frame
    first_token: int  # 第一个槽 token 的下标（append 前的序列长度）

    @property
    def num_keyframes(self) -> int:
        return len(self.pixel_frame_indices)  # 槽个数 K

    @property
    def num_tokens(self) -> int:
        return self.num_keyframes * self.tokens_per_keyframe  # 全部槽合计 token 数

    @property
    def token_slice(self) -> slice:
        return slice(self.first_token, self.first_token + self.num_tokens)  # 全部槽在序列中的连续区间


@dataclass(frozen=True)
class LatentState:  # 扩散去噪过程中一条模态流（视频或音频）的完整工作状态
    """Working state of one modality stream during diffusion denoising.

    Built by ``LatentTools.create_initial_state`` (target grid only), then grown by
    ``ConditioningItem.apply_to`` which **always appends** extra tokens at the end.
    After the denoising loop, ``clear_conditioning`` drops tokens past the target grid
    and, if a layout is present, extracts the slot content into ``generated_keyframes``.

    Field groups:
        Denoise core -- ``latent`` / ``clean_latent`` / ``denoise_mask`` / ``positions``.
        Attention -- ``attention_mask``, grown incrementally by conditioning items.
        Generated keyframes -- ``keyframes_mask`` marks single-pixel-frame tokens for the
            transformer embedding; ``generated_keyframe_layout`` locates the slots;
            ``generated_keyframes`` holds extracted slot content after clear.
        Stream flag -- ``frozen`` locks this modality so it is not denoised.

    ``keyframes_mask`` vs ``generated_keyframe_layout``: the mask tells the transformer
    *which tokens get the keyframe embedding* (first latent frame + slots). The layout
    tells post-processing *where the slots sit and which pixel times they represent*.
    Given-content keyframe conditioning is not marked and does not write a layout.

    Attributes:
        latent: Current noisy tokens being denoised. Patchified shape ``(B, T, C)``.
        denoise_mask: Per-token denoise strength, same T as ``latent``. 1 = full denoise,
            0 = hold fixed. At mask 1 the noiser lerps from ``latent`` toward noise and
            **ignores** ``clean_latent``.
        positions: Per-token RoPE coordinates. Video default ``(B, 3, T, 2)`` holding
            ``[start, end)`` bounds on (time, height, width) in pixel space.
        clean_latent: Pre-denoise / conditioning content, same shape as ``latent``.
        attention_mask: Optional ``(B, T, T)`` self-attention weights in ``[0, 1]``.
            ``None`` = full attention. Built incrementally by conditioning items.
        keyframes_mask: Optional ``(B, T, 1)`` marker, same layout as ``denoise_mask``.
            Non-zero on tokens that encode a *single pixel frame* rather than the usual
            8-frame span: the target's first latent frame (causal encoder: frame 0 covers
            1 pixel frame, later frames cover 8) plus any generated keyframe slots.
            Selects tokens that receive the learned keyframe absolute-position embedding.
            Ignored by models built without ``use_keyframes_abs_pos_embedding``.
        generated_keyframe_layout: Bookmark written when generated-keyframe slots were
            appended. ``None`` if this state has no such slots.
        generated_keyframes: Filled by ``clear_conditioning`` from the layout: denoised
            slot content as unpatchified ``(B, C, K, H, W)``, one latent frame per
            keyframe. Decode each frame as a standalone 1-frame clip -- a causal K-frame
            decode would blend slots that were never adjacent.
        frozen: When True this stream is held fixed. ``denoise_mask`` should be all zeros
            (pipeline builders enforce that). Prompt / cross-modality AdaLN noise is
            forced to 0 when the state is converted for the transformer.

    一条模态流（视频或音频）在扩散去噪中的完整工作状态。

    由 ``LatentTools.create_initial_state`` 建出（此时只有 target 网格），再由
    ``ConditioningItem.apply_to`` **一律往序列尾部 append** 额外 token。去噪结束后
    ``clear_conditioning`` 丢掉 target 网格以外的 token；若有 layout，会先把槽内容
    抽到 ``generated_keyframes``，再砍掉额外 token。

    字段分组：
        去噪本体 -- ``latent`` / ``clean_latent`` / ``denoise_mask`` / ``positions``。
        注意力 -- ``attention_mask``，由条件项逐步扩建。
        生成关键帧 -- ``keyframes_mask`` 标出「单像素帧」token 给 transformer embedding；
            ``generated_keyframe_layout`` 定位槽；``generated_keyframes`` 是清条件后抽出的槽内容。
        流级开关 -- ``frozen`` 锁死本模态，不再去噪。

    ``keyframes_mask`` 与 ``generated_keyframe_layout`` 分工不同：mask 告诉 transformer
    *哪些 token 加 keyframe embedding*（第 0 个 latent frame + 槽）；layout 告诉后处理
    *槽在序列哪一段、对应哪些像素时刻*。给定内容的关键帧条件既不标 mask，也不写 layout。

    属性：
        latent: 当前正在去噪的噪声 token。patchify 后形状 ``(B, T, C)``。
        denoise_mask: 每 token 去噪强度，T 与 ``latent`` 对齐。1 = 全去噪，0 = 锁死。
            mask=1 时 noiser 从 ``latent`` 往噪声 lerp，**忽略** ``clean_latent``。
        positions: 每 token 的 RoPE 坐标。视频默认 ``(B, 3, T, 2)``，最后一维是像素空间
            (时间, 高, 宽) 的 ``[start, end)`` 区间。
        clean_latent: 去噪前 / 条件内容，形状与 ``latent`` 相同。
        attention_mask: 可选 ``(B, T, T)`` 自注意力权重，值在 ``[0, 1]``。``None`` = 全看。
            由条件项逐步扩建。
        keyframes_mask: 可选 ``(B, T, 1)`` 标记，布局与 ``denoise_mask`` 相同。非零表示
            该 token 编码的是**单独一帧像素**，不是通常的 8 帧跨度。两类 token 会被标：
            target 第 0 个 latent frame（因果 encoder：首帧只覆盖 1 个像素帧，其余各覆盖 8），
            以及生成关键帧槽。选出要加 learned keyframe 绝对位置 embedding 的 token。
            没有 ``use_keyframes_abs_pos_embedding`` 的模型会忽略它。
        generated_keyframe_layout: 追加生成关键帧槽时写入的定位书签。无此类槽则为 ``None``。
        generated_keyframes: ``clear_conditioning`` 按 layout 抽出的去噪后槽内容，unpatchify
            后为 ``(B, C, K, H, W)``，每个关键帧一个独立 latent frame。必须按单帧 clip
            逐个 decode，不能当成 K 帧视频一次过——因果 decode 会把从未相邻的槽混在一起。
        frozen: 为 True 时本流锁死。``denoise_mask`` 应为全 0（pipeline 会强制）。
            转成 transformer 输入时，prompt / 跨模态 AdaLN 的标量噪声也强制为 0。
    """

    latent: torch.Tensor  # 当前噪声 token，patchify 后 (B, T, C)
    denoise_mask: torch.Tensor  # 每 token 去噪强度 (B, T, 1)；1 全去噪，0 锁死
    positions: torch.Tensor  # RoPE 坐标；视频默认 (B, 3, T, 2) = (时间/高/宽) 的 [start, end)
    clean_latent: torch.Tensor  # 去噪前 / 条件内容，形状同 latent；mask=1 时被 noiser 忽略
    attention_mask: torch.Tensor | None = None  # 可选 (B, T, T) 自注意力；None = 全看
    keyframes_mask: torch.Tensor | None = None  # 可选 (B, T, 1)；非零 = 单像素帧 token（首帧 + 生成槽）
    generated_keyframe_layout: GeneratedKeyframeLayout | None = None  # 槽在序列中的定位书签
    generated_keyframes: torch.Tensor | None = None  # 清条件后抽出的 (B, C, K, H, W) 槽内容
    frozen: bool = False  # True 时本流不参与去噪

    def clone(self) -> "LatentState":
        # tensor 深拷贝；layout 是不可变整数元组，浅拷贝即可
        return LatentState(
            latent=self.latent.clone(),
            denoise_mask=self.denoise_mask.clone(),
            positions=self.positions.clone(),
            clean_latent=self.clean_latent.clone(),
            attention_mask=self.attention_mask.clone() if self.attention_mask is not None else None,
            keyframes_mask=self.keyframes_mask.clone() if self.keyframes_mask is not None else None,
            generated_keyframe_layout=self.generated_keyframe_layout,
            generated_keyframes=(self.generated_keyframes.clone() if self.generated_keyframes is not None else None),
            frozen=self.frozen,
        )
