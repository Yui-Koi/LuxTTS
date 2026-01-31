# LuxTTS / ZipVoice Architecture

This document provides a high-level overview of the LuxTTS system, its core components, data flow, and main extension points. It is intended for developers who want to understand how the pieces fit together and where to modify or extend the system.

---

## 1. High-Level Overview

LuxTTS is a high-quality voice cloning and text-to-speech (TTS) system built around a distilled version of the **ZipVoice** model, paired with a custom **48 kHz Vocos vocoder**. The system supports:

- Full PyTorch inference on GPU/MPS/CPU
- CPU-optimized ONNX Runtime inference
- TensorRT-accelerated decoding on CUDA

The repository is organized roughly as:

- `zipvoice/models/`: ZipVoice, distilled models, dialog variants
- `zipvoice/tokenizer/`: tokenizers and text normalization
- `zipvoice/utils/`: feature extraction, optimization utils, schedulers, export helpers
- `zipvoice/bin/`: CLI scripts for dataset preparation, feature computation, training, inference, and export
- `zipvoice/luxvoice.py`: high-level Python API for prompt encoding and speech generation

The core TTS pipeline is:

1. **Text → Tokens → Frame-level conditions** (tokenizers + text encoder)
2. **Prompt audio → Acoustic features** (Vocos-style log-mel fbank)
3. **Flow-matching acoustic model** (ZipVoice / ZipVoiceDistill) generates features
4. **Vocoder** converts features to 48 kHz audio

---

## 2. Acoustic Feature Representation

LuxTTS operates on log-mel filterbank features computed with `VocosFbank` (see `zipvoice/utils/feature.py`):

- **Sample rate (features)**: 24 kHz
- **Filterbank**: 100 mel bins, 1024 FFT size, 256 hop length
- **Mono features**: shape `[T, 100]`
- **Stereo/dialog features**: concatenated along channel, shape `[T, 200]`

A global scaling factor is applied:

- During training: features are multiplied by `feat_scale` (default `0.1`)
- During inference: the inverse scaling is applied before vocoding

This scaling stabilizes training and improves numerical behavior for the flow-matching objective.

---

## 3. Core Modeling Components

### 3.1 ZipVoice Base Model (`zipvoice/models/zipvoice.py`)

The base **ZipVoice** model is a flow-matching acoustic model conditioned on text and (optionally) speech prompts.

Key components:

- **Text Embedding + Encoder**
  - Token IDs are embedded and passed through a `TTSZipformer` text encoder.
  - The encoder outputs frame-level text conditions aligned to the acoustic feature sequence.

- **Flow-Matching Decoder (`fm_decoder`)**
  - Another `TTSZipformer` instance is used as the flow-matching decoder.
  - Inputs at each step are the concatenation of:
    - Noisy features (current state of diffusion)
    - Text conditions
    - Speech conditions (prompt features), if used
  - These are concatenated along the feature dimension: `[B, T, 3 * feat_dim]`.

- **Objective: Continuous-Time Flow Matching**
  - The model learns a **velocity field** `v(x, t)` over a diffusion process between noise and data.
  - Loss is an MSE between predicted velocity and the analytically defined target velocity.

- **Sampling: Euler ODE Solver**
  - Sampling uses an `EulerSolver` that integrates the ODE over a schedule of time steps.
  - The solver operates on the learned velocity field `v = x_1 - x_0` and analytically
    reconstructs predicted `(x_0, x_1)` at each step before re-evaluating the linear
    mixture at the next time `t_next`.
  - Time steps can include **rational time shifts** (`t_shift`) to emphasize low-SNR
    regions and improve perceptual quality.

#### 3.1.1 Classifier-Free Guidance and Conditions

At inference time, ZipVoice is wrapped by a `DiffusionModel` that implements
**classifier-free guidance (CFG)**:

- The decoder is called on a concatenation of **conditional** and **unconditional**
  batches (2× batch size) when `guidance_scale > 0`.
- Unconditional predictions drop **text_condition**, while prompt
  **speech_condition** is treated differently depending on the timestep:
  - Early timesteps (`t <= 0.5`): both branches see the same prompt; CFG focuses on
    text vs no-text while keeping timbre.
  - Late timesteps (`t > 0.5`): unconditional branch has no prompt; CFG also
    modulates prompt influence.
- The final velocity is combined as
  `(1 + guidance_scale) * v_cond - guidance_scale * v_uncond`.

This means that for the original ZipVoice model, each ODE step with guidance incurs
roughly **2× decoder compute**.

### 3.2 ZipVoiceDistill (`zipvoice/models/zipvoice_distill.py`)

`ZipVoiceDistill` is a distilled, faster variant of `ZipVoice` designed for efficient inference.

- Extends the base ZipVoice architecture, but:
  - Uses `DistillEulerSolver`, which calls into a `DistillDiffusionModel` wrapper
    with **embedded guidance scale** rather than runtime CFG.
  - The decoder `TTSZipformer` is configured with `use_guidance_scale_embed=True`,
    so `guidance_scale` is consumed as an additional embedding instead of doubling
    the batch.
- Supports a **two-stage distillation** pipeline (see Training Pipeline):
  1. Distillation from a frozen teacher ZipVoice model.
  2. Self-teaching using an EMA version of the distilled model.

For more detail on the ODE solver, guidance behavior, and hyperparameters such as
`num_step` and `t_shift`, see `ode_solver.md`.

### 3.3 Dialog Models (`zipvoice/models/zipvoice_dialog.py`)

There are dialog-capable variants:

- **`ZipVoiceDialog`**
  - Adds **speaker-turn embeddings** and special speaker tokens.
  - Input text includes explicit speaker markers which are tokenized and used for conditioning.

- **`ZipVoiceDialogStereo`** (implemented as a variant in the dialog model)
  - Uses **stereo acoustic features** `[T, 200]` (concatenated channels).
  - Adds an **energy-based loss** on left/right channels to encourage clear turn-taking and channel separation.

---

## 4. Text Conditioning & Duration Modeling

Text conditioning ensures the model has frame-aligned text conditions matching the acoustic feature length.

- **Token Sequences**
  - Token IDs are padded to a max length in batch.
  - An embedding layer maps IDs to vectors.
  - A `TTSZipformer` text encoder produces frame-level sequences.

- **Duration Modeling & Alignment**
  - Frame length is aligned to token length via **average token duration**.
  - Utilities:
    - `prepare_avg_tokens_durations`: computes or applies average durations.
    - `get_tokens_index`: maps frame indices to token indices.
    - `make_pad_mask` and related masking utilities: handle padding and variable lengths.

- **Training vs Inference**
  - Training: text conditions are aligned using known or simulated frame lengths.
  - Inference: frame length is predicted via **length ratios** (e.g., proportional to text length) and optionally adjusted by a **speed** factor.

---

## 5. Tokenization & Text Normalization

Tokenization is handled by classes in `zipvoice/tokenizer/tokenizer.py`, with normalization utilities in `zipvoice/tokenizer/normalizer.py`.

### 5.1 EmiliaTokenizer (Default)

- Designed for mixed **Chinese/English** input with optional inline **pinyin** and tags.
- Uses a combination of:
  - `jieba` for Chinese word segmentation
  - `pypinyin` for pinyin generation
  - `espeak` for phonemization (non-Chinese)
- Normalization pipeline (see `normalizer.py`):
  - Cleans punctuation and whitespace
  - Handles case, digits, and special symbols
  - Can insert markers/tags for prosody or control

Token files map **token strings ↔ IDs**, and tokenizers provide conversions:

- Text → tokens → IDs (for model input)
- IDs → tokens → text (for debugging/inspection)

### 5.2 Other Tokenizers

- **`DialogTokenizer`**: extends the base tokenizer with **speaker tokens** and dialog-specific formatting.
- **`EspeakTokenizer`**: focuses more directly on `espeak` phoneme sequences.
- **`LibriTTSTokenizer`**: tokenization compatible with LibriTTS-style data.
- **`SimpleTokenizer`**: a character-level fallback for simple experiments.

---

## 6. Vocoder Architecture

LuxTTS uses a Vocos-based neural vocoder, exposed via `linacodec.vocoder.Vocos`.

- Original ZipVoice used a **24 kHz vocoder**.
- LuxTTS introduces a **custom 48 kHz vocoder** with configurable frequency range.
- Acoustic features (24 kHz mel) are upsampled and converted to 48 kHz waveforms.

The high-level vocoder API:

- Accepts feature sequences (scaled back from `feat_scale`).
- Produces PCM audio at 48 kHz.
- `generate_speech` exposes a flag such as `return_48k` to trade off **quality vs smoothness**.

Note: The mixing of 24 kHz features and 48 kHz output is largely implicit in current code paths; sample rate assumptions should be kept in mind when extending or integrating.

---

## 7. Inference Pipelines

### 7.1 Python API (`zipvoice/luxvoice.py`)

The main user-facing API encapsulates encoding a **prompt** and generating speech:

- **Prompt Encoding**
  - Input: reference audio (for voice cloning) and optional transcript.
  - Pipeline: audio → acoustic features → prompt embedding used as **speech condition**.

- **Speech Generation**
  - Input: text, prompt embedding, and optional settings (speed, guidance, sampling steps).
  - Output: generated 48 kHz audio.

The API supports:

- GPU / CUDA
- Apple MPS
- CPU

with appropriate device placement for both model and vocoder.

### 7.2 ONNX CPU Inference (`zipvoice/onnx_modeling.py`, `zipvoice/bin/infer_zipvoice_onnx.py`)

ONNX Runtime is used to accelerate CPU-only inference:

- Text encoder and flow-matching decoder are exported as ONNX graphs.
- Sampling is implemented in Python but wraps ONNX inference for the heavy layers.
- The ONNX pipeline mirrors the PyTorch sampling formulation (same time discretization and conditioning logic).

Note: There is **duplicated ONNX CPU inference logic** between `onnx_modeling.py` and `bin/infer_zipvoice_onnx.py`, which can increase maintenance overhead.

### 7.3 CLI Tools (`zipvoice/bin/*.py`)

Key scripts:

- `infer_zipvoice.py`: single-speaker inference with:
  - Text chunking by punctuation
  - Batching over segments
  - Optional silence removal / trimming
  - Audio concatenation for final output

- `infer_zipvoice_dialog.py`: dialog inference:
  - Handles speaker-turn tokens
  - Splits and processes turns separately or in streaming fashion
  - Manages stereo or mono dialog outputs

- `infer_zipvoice_onnx.py`: ONNX-based inference entrypoint.
- `tensorrt_export.py` + `zipvoice/utils/tensorrt.py`: TensorRT export and CUDA execution for the flow-matching decoder.

Silence removal is currently implemented with **`pydub`**, which can be heavy for large-scale or low-latency use cases.

---

## 8. Training Pipeline

Training and distillation scripts live under `zipvoice/bin/` and rely on utilities in `zipvoice/utils/`.

### 8.1 Dataset Preparation

- `prepare_dataset.py` converts TSV metadata into **Lhotse `CutSet`** manifests for audio and text.
- These manifests describe segments, paths, and supervision information used downstream.

### 8.2 Feature Computation

- `compute_fbank.py` uses `VocosFbank` to compute and store acoustic features:
  - 24 kHz log-mel features as described in Section 2.
  - Features are typically stored in a compressed format for I/O efficiency.

### 8.3 Training Scripts

- `train_zipvoice.py`: standard ZipVoice training
- `train_zipvoice_dialog.py`: dialog model training
- `train_zipvoice_dialog_stereo.py`: stereo dialog training
- `train_zipvoice_distill.py`: ZipVoiceDistill training

Common characteristics:

- **Distributed Data Parallel (DDP)** for multi-GPU training (using NCCL backend)
- Optional **mixed precision**
- **`ScaledAdam`** optimizer (see Section 9)
- **`Eden`** learning rate scheduler (`zipvoice/utils/lr_scheduler.py`)
- OOM batch scans and dynamic batch sizing where applicable
- Checkpoint averaging and diagnostic hooks (`zipvoice/utils/checkpoint.py`, `zipvoice/utils/diagnostics.py`)

### 8.4 Distillation Stages

Distillation for `ZipVoiceDistill` typically occurs in two steps:

1. **Teacher-Student Distillation**
   - Teacher: frozen, high-quality ZipVoice model.
   - Student: ZipVoiceDistill, trained to match teacher outputs under the flow-matching objective.

2. **Self-Teaching with EMA**
   - An **EMA (Exponential Moving Average)** of the student acts as a more stable teacher.
   - The student is further refined using the EMA model as target.

The dialog stereo model adds an **energy-based loss** that encourages speaker exclusivity by penalizing overlapping energy across channels.

---

## 9. Optimization & Custom Kernels

LuxTTS includes several custom modules aimed at stability and efficiency.

### 9.1 Custom Activations (SwooshL, SwooshR)

Implemented in `zipvoice/models/modules/` using **`k2` CUDA ops**, with PyTorch fallbacks when `k2` is unavailable.

- Provide smoother, more stable activation behavior than standard ReLU/GELU in the Zipformer context.
- Fallback ensures the model can still run on CPU-only or environments without `k2`.

### 9.2 Balancer Module

The **`Balancer`** modifies gradients to enforce per-channel constraints:

- Keeps a target proportion of **positive activations** per channel.
- Controls RMS magnitude of activations.
- Parameters are scheduled over training with **`ScheduledFloat`** (see below) to gradually apply constraints.

This acts as a form of learned regularization on internal activations.

### 9.3 Whiten Module

`Whiten` performs whitening of intermediate activations:

- Removes correlations between channels.
- Improves optimization stability and may allow higher learning rates.

### 9.4 BiasNorm

A lightweight normalization alternative to LayerNorm:

- Learnable **bias + scale** per channel.
- Lower overhead and better suited to some Zipformer blocks than full LayerNorm.

### 9.5 ScaledAdam (`zipvoice/utils/optim.py`)

A variant of Adam tailored for large models and batched parameter updates:

- Scales parameter updates based on **RMS norms** of parameters.
- Supports **batched parameter updates** to reduce kernel launch overhead.
- Includes gradient clipping.

### 9.6 ScheduledFloat (`zipvoice/utils/common.py` or related)

A utility for **piecewise scheduling** of scalar hyperparameters based on global batch count:

- Controls dropout rates
- Tunes Balancer parameters
- Can be used for other floats that need time-dependent behavior

A global batch counter is maintained during training so all scheduled parameters evolve consistently.

---

## 10. Known Limitations & Fragile Areas

When extending or refactoring, be aware of the following issues:

- **`load_models_cpu` model path handling**
  - Currently ignores the passed `model_path` and always downloads from HuggingFace.
  - This limits offline or local-only usage and makes overrides harder.

- **Fragile return-value unpacking in `LuxTTS.encode_prompt`**
  - Relies on dictionary `.values()` ordering when unpacking multiple values.
  - This is not guaranteed and should be replaced with explicit key-based access.

- **Hidden speed scaling**
  - `generate` and `generate_cpu` apply an internal `speed = speed * 1.3` factor.
  - This is undocumented and may confuse users when tuning speed.

- **Device handling in `load_models_gpu`**
  - Mixes `torch.device(device, 0)` with `pipeline(device=device)` and manual placement.
  - This can be brittle across multi-GPU or non-standard device setups.

- **ONNX code duplication**
  - ONNX inference logic is partially duplicated between `onnx_modeling.py` and `bin/infer_zipvoice_onnx.py`.
  - Changes to sampling or model structure must be mirrored in both places.

- **Silence removal with `pydub`**
  - `pydub`-based audio segmentation and conversion can be slow and memory-heavy for large batches or low-latency applications.

- **Implicit sample rate assumptions**
  - 24 kHz features feeding a 48 kHz vocoder are assumed implicitly.
  - When integrating with external tools, ensure sample rate conversions are handled correctly.

- **DDP using NCCL-only backend**
  - Training assumes NCCL for distributed runs.
  - CPU-only clusters or other backends are not currently supported out of the box.

---

## 11. Extension & Customization Guidelines

This section summarizes recommended patterns for extending LuxTTS.

### 11.1 New Vocoder or Feature Extractor

To change acoustic features or vocoder:

1. Implement a new feature extractor in `zipvoice/utils/feature.py` (or a new module):
   - Define how to compute features (sample rate, mel bins, FFT, hop size).
   - Ensure you handle `feat_scale` consistently.
2. Update training configs and scripts to use the new extractor.
3. Update inference code paths (`luxvoice.py`, `zipvoice/utils/infer.py`, CLIs) to load and apply the new extractor.
4. Integrate a corresponding vocoder (or adapt the existing Vocos interface) to handle the new features and sample rate.

### 11.2 New Samplers or Flow-Matching Schedules

To change the sampling strategy or number of flow-matching steps:

1. Implement a new solver class (similar to `EulerSolver`).
2. Plug it into the sampling code paths in the model (ZipVoice/ZipVoiceDistill) or inference utilities.
3. Optionally expose new sampler parameters (step counts, time shifts) via CLI and Python API.

### 11.3 New Tokenizers

To add a tokenizer:

1. Implement a new `Tokenizer`-compatible class in `zipvoice/tokenizer/tokenizer.py`.
2. Define mappings between text, tokens, and IDs, and provide a token file.
3. Wire it into CLI options (e.g., `--tokenizer` or config entries).
4. Ensure normalization in `normalizer.py` is appropriate or add new rules as needed.

### 11.4 Modifying Flow-Matching Objectives

To experiment with new objectives:

1. Modify `ZipVoice.forward` (and related methods) in `zipvoice/models/zipvoice.py`.
2. Adjust or add masking utilities such as `condition_time_mask` and related functions.
3. Update loss computation in training scripts to include new terms or weights.
4. For dialog stereo, extend or replace the energy-based loss to support new behaviors.

### 11.5 Export & Deployment Considerations

- Before exporting to ONNX, use `convert_scaled_to_non_scaled` in `zipvoice/utils/scaling_converter.py` (or equivalent) to ensure parameters are in a compatible format.
- Maintain the global batch counter for training so:
  - Schedulers (`ScheduledFloat`, learning rate schedulers) behave correctly.
  - EMA and Balancer modules receive the right time signals.
- When adding new modules that need to be exported:
  - Ensure they are ONNX-compatible where possible.
  - If not, provide custom wrappers or fallbacks in ONNX inference paths.

---

## 12. Summary

LuxTTS combines a Zipformer-based flow-matching acoustic model (ZipVoice / ZipVoiceDistill) with a high-fidelity 48 kHz Vocos vocoder, driven by flexible tokenizers and robust training utilities. Understanding the separation between:

- **Text processing and tokenization**
- **Acoustic modeling (flow matching)**
- **Vocoder synthesis**
- **Training, export, and inference pipelines**

is key to safely extending or modifying the system. When in doubt, follow the patterns in existing training and inference scripts and keep an eye on the known fragile areas listed above.
