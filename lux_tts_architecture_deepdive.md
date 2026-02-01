# LuxTTS / ZipVoice TTS Architecture – Deep Dive

Author: Lumen (for Kiwi)

This document is an end-to-end architectural analysis of the LuxTTS repository.
It explains **what is being modeled**, **how the pieces connect**, and
**where the numerical and systems choices live**.

LuxTTS is a thin but opinionated layer on top of Xiaomi’s ZipVoice stack.
Almost all heavy lifting (modeling, training, schedulers) lives in
`zipvoice/`; LuxTTS adds:

- A **distilled 4-step ZipVoiceDistill** acoustic model.
- A **custom Vocos-based 48 kHz vocoder** package in the Hugging Face repo.
- A **LuxTTS** Python wrapper that wires **prompt ASR → features → flow-matching → vocoder**.
- **ONNX and TensorRT** export/inference paths for fast CPU/GPU deployment.

At a high level, the TTS pipeline is:

1. **Prompt audio** → 24 kHz mel features (VocosFbank) → prompt feature tensor.
2. **Prompt audio** (16 kHz) → Whisper ASR → prompt text.
3. **Target text** → phoneme tokens (EmiliaTokenizer) → frame-aligned text conditions.
4. **Flow-matching acoustic model** (ZipVoiceDistill) generates new mel features,
   conditioned on both text and prompt features.
5. **Vocoder** converts 24 kHz mel features to **48 kHz** audio (LuxTTS variant),
   with configurable bandwidth (e.g., `freq_range = 12000`).

---

## 1. Top-level runtime API: `LuxTTS`

**File:** `zipvoice/luxvoice.py`  
**Support code:** `zipvoice/modeling_utils.py`, `zipvoice/onnx_modeling.py`

### 1.1. Construction

```python
class LuxTTS:
    def __init__(self, model_path='YatharthS/LuxTTS', device='cuda', threads=4):
        ...
        if device == 'cpu':
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_cpu(...)
        else:
            model, feature_extractor, vocos, tokenizer, transcriber = load_models_gpu(...)
        ...
        self.vocos.freq_range = 12000
```

Key points:

- **Model weights & assets** are fetched from the Hugging Face repo
  `YatharthS/LuxTTS` by default.
- **Device selection:**
  - If `device='cuda'` but CUDA is unavailable, falls back to MPS, then CPU.
  - For CPU, it uses ONNX Runtime via `OnnxModel`; for GPU/MPS it uses
    PyTorch `ZipVoiceDistill`.
- `load_models_gpu` / `load_models_cpu` (see below) also construct:
  - **Tokenizer:** `EmiliaTokenizer` with `tokens.txt` from the HF repo.
  - **Feature extractor:** `VocosFbank` (24 kHz, 100-bin log-mel).
  - **Vocoder:** `linacodec.vocoder.vocos.Vocos` with a custom 48 kHz
    configuration bundled in the repo.
  - **Transcriber:** a Whisper ASR pipeline from `transformers.pipeline`.
- The vocoder is customized by setting `self.vocos.freq_range = 12000`, which
  adjusts bandwidth.

### 1.2. Prompt encoding

```python
@torch.inference_mode
def process_audio(audio, transcriber, tokenizer, feature_extractor, device,
                  target_rms=0.1, duration=4, feat_scale=0.1):
    prompt_wav, sr = librosa.load(audio, sr=24000, duration=duration)
    prompt_wav2, sr = librosa.load(audio, sr=16000, duration=duration)
    prompt_text = transcriber(prompt_wav2)["text"]
    ...
    prompt_wav = torch.from_numpy(prompt_wav).unsqueeze(0)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)
    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=24000)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
    return prompt_tokens, prompt_features_lens, prompt_features, prompt_rms
```

`LuxTTS.encode_prompt` is just a light wrapper:

```python
def encode_prompt(self, prompt_audio, duration=5, rms=0.001):
    prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = \
        process_audio(...)
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_features_lens": prompt_features_lens,
        "prompt_features": prompt_features,
        "prompt_rms": prompt_rms,
    }
```

Semantics:

- **Audio is decoded twice**:
  - 24 kHz for feature extraction.
  - 16 kHz for Whisper ASR to extract the prompt transcript.
- **RMS normalization** (`rms_norm`) ensures the prompt waveform is lifted up to
  `target_rms` if it was quieter, and returns the original RMS for later
  volume matching.
- **Feature extractor** (`VocosFbank`) computes **24 kHz, 100-bin log-mel**
  features, then they are scaled by `feat_scale = 0.1` for numerical stability.
- The prompt text is tokenized with `EmiliaTokenizer` into phone-like tokens.

### 1.3. Generation

```python
def generate_speech(self, text, encode_dict,
                    num_steps=4, guidance_scale=3.0, t_shift=0.5,
                    speed=1.0, return_smooth=False):
    prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = \
        encode_dict.values()

    self.vocos.return_48k = not return_smooth

    if self.device == 'cpu':
        final_wav = generate_cpu(...)
    else:
        final_wav = generate(...)
    return final_wav.cpu()
```

Important details:

- **Shape conventions**:
  - `prompt_features`: `(B=1, T_prompt, feat_dim=100)`.
  - `prompt_features_lens`: `(1,)`.
  - `prompt_tokens`: list `[ [ids...] ]`.
- **Speed scaling:** both `generate` and `generate_cpu` re-scale speed as

  ```python
  speed = speed * 1.3
  ```

  so the user-facing `speed` parameter is not the actual internal ratio.
- **return_smooth:** controls `vocos.return_48k`:
  - `False` (default): return 48 kHz high-bandwidth output.
  - `True`: return (typically) 24 kHz “smooth” output.

`generate` (GPU path) performs:

1. Text tokenization with `EmiliaTokenizer`.
2. Call into `ZipVoiceDistill.sample` with the prompt as **speech condition**.
3. Undo feature scaling and call the vocoder.
4. RMS-match the output to the prompt’s original RMS if the prompt was quieter
   than `target_rms`.

`generate_cpu` (ONNX path) mirrors the same steps via `OnnxModel.sample`.

---

## 2. Acoustic feature representation & vocoder

### 2.1. VocosFbank features

**File:** `zipvoice/utils/feature.py`

```python
@dataclass
class VocosFbankConfig:
    sampling_rate: int = 24000
    n_mels: int = 100
    n_fft: int = 1024
    hop_length: int = 256
```

`VocosFbank`:

- Uses `torchaudio.transforms.MelSpectrogram` with `power=1` (magnitude), then
  applies `log()` after clamping to `1e-7`.
- Returns shape `(T, feat_dim)` where `feat_dim = n_mels * num_channels`.
- Enforces **24 kHz** sampling rate.
- Uses Lhotse’s `compute_num_frames` to ensure time alignment, padding or
  trimming if necessary.

In both training and inference, features are multiplied by `feat_scale = 0.1`.
At inference time, this scaling is undone before vocoding.

### 2.2. Vocoder choices

There are **two vocoder families** in the repo:

1. **Original ZipVoice vocoder** (`zipvoice/bin/infer_zipvoice.py`):
   - Uses `vocos.Vocos.from_pretrained("charactr/vocos-mel-24khz")` if no
     local checkpoint is provided.
   - Operates at 24 kHz.

2. **LuxTTS vocoder** (`zipvoice/modeling_utils.py`):
   - Uses `linacodec.vocoder.vocos.Vocos.from_hparams` with
     `config.yaml` and `vocos.bin` from `YatharthS/LuxTTS`.
   - Produces **48 kHz audio**, with `self.vocos.freq_range` set to 12 kHz.
   - Has a flag `return_48k` used to choose between full-band 48 kHz and
     downsampled 24 kHz outputs.

The acoustic model **always operates on 24 kHz mel features**. The 48 kHz
behavior is entirely in the vocoder’s upsampling stack.

---

## 3. Text processing & tokenization

**File:** `zipvoice/tokenizer/tokenizer.py`

### 3.1. EmiliaTokenizer (default for LuxTTS)

Emilia is a **phone-like tokenizer** designed for Chinese/English and pinyin.

Pipeline per text string:

1. **Normalization:**
   - `map_punctuations` normalizes CJK punctuation to ASCII equivalents.
   - English and Chinese text normalizers in `normalizer.py` handle case,
     whitespace, digits, etc.
2. **Segmentation by language/type:**
   - `get_segment` classifies each character (or `<...>` / `[...]` unit) as
     `zh`, `en`, `pinyin`, or `other` and groups them into segments.
   - `split_segments` further splits segments when special units `<...>`
     (pinyin) or `[...]` (tags) appear.
3. **Language-specific tokenization:**
   - `tokenize_ZH`:
     - Uses `jieba` for word segmentation.
     - `pypinyin` (tone3 style) to generate pinyin with tone marks.
     - Splits pinyin into initials and finals via `to_initials` and
       `to_finals_tone3`; encodes them as separate tokens (initials get a
       trailing `0`).
   - `tokenize_EN`:
     - Normalizes with `EnglishTextNormalizer`.
     - Uses `phonemize_espeak` with `en-us` to get phoneme tokens.
   - `tokenize_pinyin`:
     - For `<pinyin>` segments, verifies correctness and splits to
       initials/finals as above.
   - `tag` segments (inside `[]`) are passed through as whole tokens.
4. **Mapping to IDs:**
   - `token2id` loaded from `tokens.txt`.
   - OOV tokens are **silently skipped** with a debug log.

Shapes:

- Tokenization returns `List[List[str]]` for texts.
- `texts_to_token_ids` returns `List[List[int]]` of the same structure.
- `pad_id` and `vocab_size` are read from `tokens.txt`.

### 3.2. Dialog tokenization

`DialogTokenizer(EmiliaTokenizer)` differs by:

- Defining `spk_a_id`, `spk_b_id` based on token IDs for `[S1]`, `[S2]`.
- Overriding `preprocess_text` to:
  - Remove whitespace around `[S1]` / `[S2]` via a regex.
  - Then call `map_punctuations`.

Dialog models rely on these explicit speaker tags in text.

### 3.3. Other tokenizers

- `SimpleTokenizer`: character-level, no normalization.
- `EspeakTokenizer`: phonemizes arbitrary text using `espeak-ng` for a given
  language (via `piper_phonemize`).
- `LibriTTSTokenizer`: supports `char`, `phone`, and `bpe` modes with
  `espnet_tts_frontend` pre-processing and optional SentencePiece.

These are used primarily for training pipelines and alternative datasets.

---

## 4. ZipVoice core model family

**Files:**

- `zipvoice/models/zipvoice.py` – base single-speaker model.
- `zipvoice/models/zipvoice_distill.py` – distilled, fast model.
- `zipvoice/models/zipvoice_dialog.py` – dialog and stereo dialog variants.
- `zipvoice/models/modules/zipformer.py` – Zipformer backbone.
- `zipvoice/models/modules/zipformer_two_stream.py` – stereo/two-stream variant.
- `zipvoice/models/modules/solver.py` – flow-matching sampler & CFG wrapper.

### 4.1. TTSZipformer backbone

`TTSZipformer` is the main sequence model used as both **text encoder** and
**flow-matching decoder**.

Key traits (from `zipformer.py`):

- **U-Net–like multi-stack encoder**:
  - `downsampling_factor`: tuple like `(1, 2, 4, 2, 1)`; enforced to be
    symmetric (U-Net pattern).
  - For each factor, there is a `Zipformer2Encoder` (possibly wrapped by
    `DownsampledZipformer2Encoder`).
- **Relational multi-head attention + conv** per layer:
  - `Zipformer2EncoderLayer` includes:
    - `RelPositionMultiheadAttentionWeights` (separate Q/K/V logic).
    - Two `SelfAttention` modules using these weights.
    - Three `FeedforwardModule`s with different roles.
    - Two `ConvolutionModule`s (depthwise 1D conv with GLU gating).
    - `NonlinAttention` module (attention-like nonlinearity over attended
      features).
    - Multiple `Balancer` and `Whiten` modules for gradient and activation
      control.
  - Extensive use of `ScheduledFloat` to schedule skip rates and dropout over
    training batches.
- **Time- and guidance-scale embeddings:**

  ```python
  self.use_time_embed = use_time_embed
  self.use_guidance_scale_embed = use_guidance_scale_embed

  if use_time_embed:
      self.time_embed = nn.Sequential(... SwooshR() ...)

  if use_guidance_scale_embed:
      self.guidance_scale_embed = ScaledLinear(guidance_scale_embed_dim,
                                               time_embed_dim, bias=False)
  ```

  - In the forward pass, `t` and (for distilled models) `guidance_scale` are
    mapped via sinusoidal `timestep_embedding` then projected.
  - This `time_emb` is added inside each layer.

- **Input/output projections:**

  - `in_proj: Linear(in_dim → encoder_dim)`.
  - `out_proj: Linear(encoder_dim → out_dim)`.
  - Forward shape: input `(B, T, D) → (B, T, out_dim)`.

This Zipformer stack is designed to be numerically robust (Balancers,
Whiteners), dropout-scheduled, and ONNX/TensorRT exportable.

### 4.2. ZipVoice: base flow-matching acoustic model

**File:** `zipvoice/models/zipvoice.py`

#### 4.2.1. Submodules

- `self.fm_decoder: TTSZipformer` with:
  - `in_dim = feat_dim * 3` (noisy features + text condition + speech condition).
  - `out_dim = feat_dim` (velocity vector field in feature space).
  - `use_time_embed = True`, `time_embed_dim` provided.
- `self.text_encoder: TTSZipformer` with:
  - `in_dim = text_embed_dim`.
  - `out_dim = feat_dim` (text condition dimension matches feature dim).
  - `use_time_embed = False`.
- `self.embed: nn.Embedding(vocab_size, text_embed_dim)` for token IDs.
- `self.solver: EulerSolver(self, func_name="forward_fm_decoder")` for
  sampling.

#### 4.2.2. Text conditioning & alignment

1. **Embedding & encoding** (`forward_text_embed`):

   - Pads token sequences with `pad_labels` to `(B, S_max)`.
   - Embeds via `nn.Embedding` to `(B, S_max, text_embed_dim)`.
   - Creates `tokens_padding_mask` via `make_pad_mask(tokens_lens, S_max)`.
   - Runs `self.text_encoder(x=embed, t=None, padding_mask=tokens_padding_mask)`.

2. **Frame-level alignment** (`forward_text_condition`):

   - `features_lens`: target frame counts.
   - `prepare_avg_tokens_durations(features_lens, tokens_lens)` returns token-level
     durations (integer, per batch element).
   - `get_tokens_index(durations, num_frames)` expands durations into a
     `(B, T)` index mapping.
   - Gathers from encoder outputs with `torch.gather` to produce
     **text_condition** `(B, T, feat_dim)`.

This alignment is used both at **training** and **inference** (with slightly
different length logic).

#### 4.2.3. Flow-matching training objective

**Training forward:** `ZipVoice.forward`

Input shapes:

- `features`: `(B, T, F)` – ground-truth mel features.
- `features_lens`: `(B,)` – lengths in frames.
- `noise`: `(B, T, F)` – sampled from `N(0, I)`.
- `t`: `(B, 1, 1)` – uniform in (0, 1) at training time.

Steps:

1. **Text conditioning:**

   ```python
   text_condition, padding_mask = forward_text_train(tokens, features_lens)
   ```

2. **Speech condition masking:**

   ```python
   speech_condition_mask = condition_time_mask(..., mask_percent=(0.7, 1.0))
   speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, features)
   ```

   - This zeroes out a random contiguous time span (70–100% of sequence length)
     to encourage robustness to partial prompt conditions.

3. **Optional text condition dropout:**

   ```python
   if condition_drop_ratio > 0:
       drop_mask = (torch.rand(B,1,1, device) > condition_drop_ratio)
       text_condition = text_condition * drop_mask
   ```

4. **Flow-matching state & target:**

   ```python
   xt = features * t + noise * (1 - t)   # x_t
   ut = features - noise                 # v = x1 - x0
   ```

5. **Velocity prediction:**

   ```python
   vt = forward_fm_decoder(t, xt, text_condition, speech_condition, padding_mask)
   ```

   `forward_fm_decoder` concatenates `xt`, `text_condition`, `speech_condition`
   along the feature axis to shape `(B, T, 3F)` and calls `fm_decoder` with t.

6. **Loss and masking:**

   ```python
   loss_mask = speech_condition_mask & (~padding_mask)
   fm_loss = mean((vt[loss_mask] - ut[loss_mask]) ** 2)
   ```

So the acoustic model learns a **time-independent velocity field**
`v ≈ x1 − x0` over straight-line paths from noise to data.

#### 4.2.4. Inference sampling (`ZipVoice.sample`)

**Initial state:**

```python
x0 = torch.randn(B, num_frames, feat_dim, device)
```

**Conditions:**

- `text_condition`: `(B, T_full, F)` from prompt+target tokens.
- `speech_condition`: `(B, T_full, F)` from prompt mel features, padded.
- `padding_mask`: `(B, T_full)` telling which frames are real.

Then:

```python
x1 = self.solver.sample(
    x=x0,
    text_condition=text_condition,
    speech_condition=speech_condition,
    padding_mask=padding_mask,
    num_step=num_step,
    guidance_scale=guidance_scale,
    t_shift=t_shift,
)
```

Where `EulerSolver.sample` implements the deterministic
**anchor-based probability flow update** (see section 5).

Finally, `ZipVoice.sample` splits:

- **Prompt portion** vs **generated portion** based on `prompt_features_lens`.
- Returns both the generated segment without prompt frames and the regenerated
  prompt portion for convenience.

#### 4.2.5. Intermediate sampling

`ZipVoice.sample_intermediate` exposes:

```python
x_t_end = self.solver.sample(
    x=noise,
    text_condition=text_condition,
    speech_condition=speech_condition,
    padding_mask=padding_mask,
    num_step=num_step,
    guidance_scale=guidance_scale,
    t_start=t_start,
    t_end=t_end,
)
```

This is used heavily in **distillation** (teacher-student training) to compare
intermediate flows.

### 4.3. ZipVoiceDistill

**File:** `zipvoice/models/zipvoice_distill.py`

`ZipVoiceDistill(ZipVoice)` changes two things:

1. **fm_decoder:** uses `TTSZipformer` with `use_guidance_scale_embed=True`,
   so the **guidance scale** is embedded as an additional input into the
   time embedding pipeline.
2. **solver:** uses `DistillEulerSolver`, which wraps `DistillDiffusionModel`
   instead of `DiffusionModel`.

`DistillDiffusionModel` simply passes the guidance scale straight through to
`forward_fm_decoder`, instead of implementing classifier-free guidance in
Python by doubling the batch.

It also overrides `forward` to delegate to `sample_intermediate`, so during
training the **teacher-student objective** is expressed over endpoint
trajectories `x_t_start → x_t_end` under various `t_start`, `t_end` pairs.

### 4.4. Dialog and stereo models

**File:** `zipvoice/models/zipvoice_dialog.py`

#### 4.4.1. ZipVoiceDialog

- Inherits `ZipVoice`.
- Adds speaker embeddings:

  ```python
  self.spk_embed = nn.Embedding(2, feat_dim)
  ```

- Uses `DialogTokenizer`, where `[S1]` and `[S2]` are tagged.
- `extract_spk_indices` scans the token IDs to determine which tokens belong
  to speaker A/B (using cumulative counts of special IDs  `spk_a_id`, `spk_b_id`).
- `forward_text_embed`:
  - Computes base embeddings & runs `text_encoder`.
  - Adds speaker embeddings at token positions for S1/S2.

Training objective is same flow-matching loss, except:

- `condition_time_mask_suffix` masks from the **end** of sequences (suffix
  masking) instead of random segments.

#### 4.4.2. ZipVoiceDialogStereo

- Inherits `ZipVoiceDialog`.
- Replaces `fm_decoder` with `TTSZipformerTwoStream`:

  ```python
  self.fm_decoder = TTSZipformerTwoStream(
      in_dim=(feat_dim * 5, feat_dim * 3),
      out_dim=(feat_dim * 2, feat_dim),
      ...
  )
  ```

- The idea is to support **two-channel (stereo) features** and to produce
  **two independent fbank streams** (left and right channels) while sharing
  some information.

Additional loss term: **speaker-exclusive energy loss**.

- After computing `vt`, they form a “target” fbank estimate:

  ```python
  target = xt + vt * (1 - t)
  fbank_1 = target[:, :, :feat_dim]
  fbank_2 = target[:, :, feat_dim:]
  ```

- `energy_based_loss(fbank1, fbank2, gt_fbank)` computes:
  - Per-channel energies via mean over mel bins.
  - Adaptive thresholds derived from ground-truth energy distribution.
  - Penalty when **both** channels exceed the per-frame threshold (both
    speakers talking at once) by multiplying their excess energies.
- Final loss:

  ```python
  loss = fm_loss + se_weight * energy_loss
  ```

This encourages clear speaker separation between left and right channels.

---

## 5. Flow-matching sampler & guidance

**File:** `zipvoice/models/modules/solver.py`

The sampler logic is used in three places:

- `ZipVoice.solver` (full model, Python CFG).
- `ZipVoiceDistill.solver` (distilled model, CFG inside fm_decoder).
- `onnx_modeling.OnnxModel.sample` (ONNX CPU path reimplements identical math).

### 5.1. Time schedule

```python
def get_time_steps(t_start=0.0, t_end=1.0, num_step=10, t_shift=1.0, device=...):
    timesteps = torch.linspace(t_start, t_end, num_step + 1, device=device)
    if t_shift == 1.0:
        return timesteps
    inv_s = 1.0 / t_shift
    denom = torch.add(inv_s, timesteps, alpha=1.0 - inv_s)
    return timesteps.div_(denom)
```

This represents:

> `t' = t_shift * t / (1 + (t_shift - 1) * t)`.

- If `t_shift < 1`, timesteps are concentrated near 0 (noisy region).
- Training seems to assume **uniform t** in [0,1]; inference applies this
  monotone reparameterization.

### 5.2. Classifier-free guidance wrapper – non-distilled models

`DiffusionModel` wraps a model method `func_name` (e.g., `forward_fm_decoder`).

Important behavior:

- If `guidance_scale == 0`, simply calls the model once.
- If `guidance_scale != 0`:
  - **Batch duplication:** `x`, `padding_mask` are concatenated along batch.
  - **Text condition:** uncond branch has all zeros.
  - **Speech condition:**
    - If `t > 0.5`: uncond branch has zeros; speech prompt only on cond branch.
    - If `t <= 0.5`: both branches share speech_condition and `guidance_scale`
      is doubled.
  - The model is called once, output is chunked into `(uncond, cond)` and CFG is:

    ```python
    res = (1 + w) * data_cond - w * data_uncond
    ```

This is **velocity-space CFG** with a **time-dependent treatment** of speech
prompt: near data (high t), uncond branch has no prompt; near noise (low t),
prompt is present in both branches but guidance is stronger.

### 5.3. Distilled CFG wrapper

`DistillDiffusionModel.forward` simply calls the model with a scalar/tensor
`guidance_scale` argument; the internal `TTSZipformer` uses
`use_guidance_scale_embed=True` to embed this into the time embedding.

### 5.4. EulerSolver – anchor-based probability flow

For each step:

```python
for step in range(num_step):
    t_cur = timesteps[step]
    t_next = timesteps[step + 1]

    v = model(t=t_cur, x=x, text_condition=..., speech_condition=..., ...)

    # Flow-matching identities for straight line x_t = (1-t)*x0 + t*x1
    # with v = x1 - x0:
    x_1_pred = x + (1.0 - t_cur) * v
    x_0_pred = x - t_cur * v

    if step < num_step - 1:
        x = (1.0 - t_next) * x_0_pred + t_next * x_1_pred
    else:
        x = x_1_pred
```

Key semantic points:

- Training target is **time-independent** `v = x1 − x0` (features – noise).
- At any t, we can solve exactly for `(x0, x1)` from `(x_t, v)` if the straight-line
  assumption holds.
- The update to `t_next` is **exact under that model**, not an Euler
  approximation.
- There is **no noise injection** inside the loop: all stochasticity is from
  initial `x0 ~ N(0, I)`.

This is why renaming or refactoring this as “Euler” in the ODE sense would be
misleading; it is more like an **exact projection along a learned line**.

ONNX CPU path (`onnx_modeling.sample`) replicates this logic, but the
classifier-free guidance is moved into `OnnxFlowMatchingModel` exported to
ONNX.

---

## 6. ONNX and TensorRT deployment

### 6.1. ONNX export

**File:** `zipvoice/bin/onnx_export.py`

Two wrapper modules are defined for export:

1. `OnnxTextModel` – wraps `ZipVoice` text path:

   - Inputs: `tokens`, `prompt_tokens`, `prompt_features_len`, `speed`.
   - Concatenates prompt and target tokens, runs `embed` + `text_encoder`.
   - Reconstructs text_condition to match frame length, performing an average
     duration calculation and expansion inside the ONNX graph.

2. `OnnxFlowMatchingModel` – wraps the flow-matching decoder:

   - Non-distilled variant **implements CFG logic inside ONNX** by doubling x,
     zeroing conditions, and recombining outputs.
   - Distilled variant passes `guidance_scale` directly.

Before export, `convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)`
removes certain scaling constructs (Balancer, etc.) to yield more standard
weights.

The exported ONNX models are then optionally quantized (`onnxruntime`)
into int8-weight variants.

### 6.2. ONNX CPU inference

**File:** `zipvoice/onnx_modeling.py`

`OnnxModel` manages two `onnxruntime.InferenceSession`s:

- `text_encoder.onnx`.
- `fm_decoder.onnx`.

`run_text_encoder` and `run_fm_decoder` handle numpy ↔ torch conversion and
dynamic axes. The sampling logic is almost identical to `EulerSolver.sample`,
except that:

- Padding masks are not passed into ONNX path; the ONNX fm_decoder does not
  receive `padding_mask`.
- The CPU path uses a default `t_shift` of `0.9` in `generate_cpu`, favoring
  slightly different emphasis across t.

### 6.3. TensorRT acceleration

**Files:** `zipvoice/bin/tensorrt_export.py`, `zipvoice/utils/tensorrt.py`

- `tensorrt_export.py`:
  - Loads a ZipVoice/ZipVoiceDistill model.
  - Exports only **fm_decoder** to ONNX with a carefully constructed input
    signature (concatenated `x`, `text_condition`, `speech_condition` already
    applied for non-distilled) and dynamic batch/length axes.
  - Builds a TensorRT engine (`.plan`) with FP16 precision.
- `TrtContextWrapper` wraps the TensorRT engine to look like a PyTorch module:
  - Maintains a pool of TensorRT contexts to allow concurrent inference.
  - In `__call__`, sets input shapes and tensor addresses, calls
    `execute_async_v3`, and wraps output as a torch tensor.
  - Hardcodes `self.feat_dim = 100` (feature dimension).
- `load_trt(model, trt_model_path)` deletes `model.fm_decoder` and replaces it
  with `TrtContextWrapper`, so the rest of the model and sampler logic remains
  unchanged.

---

## 7. Training architecture

The training side is mostly ZipVoice’s original code. Key training scripts:

- `zipvoice/bin/train_zipvoice.py` – base ZipVoice flow-matching training.
- `zipvoice/bin/train_zipvoice_dialog.py` – dialog version.
- `zipvoice/bin/train_zipvoice_dialog_stereo.py` – stereo dialog.
- `zipvoice/bin/train_zipvoice_distill.py` – two-stage distillation to
  ZipVoiceDistill.

They share a common pattern:

- Data handling with **Lhotse** (`TtsDataModule` produces `CutSet`s and
  PyTorch dataloaders).
- **DDP + NCCL** training when `--world-size > 1`.
- **ScaledAdam** optimizer (`zipvoice/utils/optim.py`) with
  `get_parameter_groups_with_lrs` for per-module learning-rate scaling.
- Custom `Eden` or `FixedLRScheduler` LR schedules.
- Activation diagnostics (`zipvoice/utils/diagnostics.py`) and inf-check hooks.
- Use of `ScheduledFloat` for dropout/skip-rate schedules.
- Feature scaling by `feat_scale` on input features and unscaled in losses.

### 7.1. Base ZipVoice training

`train_zipvoice.compute_fbank_loss`:

- Draws Gaussian noise, samples t.
- Computes the same `xt`, `ut`, and flow-matching loss described above.
- For validation, t is deterministic across batch indices for coverage.

`train_one_epoch` then handles:

- Autocast fp16 training.
- Dynamic grad scaling.
- Periodic validation and checkpointing.
- Optional OOM scanning to find worst batches.

### 7.2. Distillation training

**File:** `zipvoice/bin/train_zipvoice_distill.py`

Two phases controlled by `--distill-stage {first,second}`:

1. **First stage** – teacher is full ZipVoice:
   - Teacher: `ZipVoice` loaded from a high-quality checkpoint.
   - Student: `ZipVoiceDistill` initialized from the same or similar weights.
   - For each batch:
     - Sample a single t value per batch `t_value` and t deltas
       `t_delta_fix`, `t_delta_ema`.
     - `xt` is built from features and noise at `t_value`.
     - Teacher runs **two** single-step flows to get `target_x1` at
       `t_dest = t_value + t_delta_fix + t_delta_ema`.
     - Student is asked to match `target_x1` in a direct single step from `xt`
       to `t_dest`, comparing velocities `(x1 - xt)/(t_dest - t)` over
       masked regions.
   - Reference loss (`ref_loss`) compares student’s velocities to the ideal
     `features - noise` target to monitor closeness to original objective.

2. **Second stage** – EMA self-teaching:
   - Teacher model is an **EMA copy of the student** (`ema()` with decay).
   - The same step-structure is used, but teacher and student share
     architecture.
   - Only the `fm_decoder` parameters are trainable; others are frozen.

This distillation sharpens the student’s ability to take **fewer flow
steps** while approximating the high-step teacher.

---

## 8. Supporting utilities & design choices

### 8.1. Scaling & numerical stabilization

**File:** `zipvoice/models/modules/scaling.py`

Core components:

- `ScheduledFloat` – piecewise-linear scalars driven by a global `batch_count`.
- `Balancer` – modifies gradients to encourage channel-wise constraints on:
  - Proportion of positive activations.
  - RMS magnitude.
- `Whiten` – group whitening of activations with controlled impact.
- `BiasNorm` – a cheaper, bias-aware alternative to LayerNorm.
- `ActivationDropoutAndLinear` + **SwooshL/SwooshR** activations – fused
  activation+dropout+linear layers, with custom CUDA implementations (via k2)
  and PyTorch fallbacks.

These are used inside Zipformer to maintain numerical stability and training
signal quality, and are mostly transparent in inference.

`scaling_converter.convert_scaled_to_non_scaled` is used before ONNX/TensorRT
export to convert these scaled parametrizations into standard forms
(e.g. folding scaling into Linear weights).

### 8.2. Common utilities

**File:** `zipvoice/utils/common.py`

Features that matter architecturally:

- `prepare_avg_tokens_durations`, `get_tokens_index`, `make_pad_mask` – core
  for text/feature alignment.
- `condition_time_mask`, `condition_time_mask_suffix` – used to construct
  speech_condition masks for training.
- `torch_autocast`, `create_grad_scaler` – version-agnostic AMP wrappers.
- `set_batch_count` – walks the module tree, setting `batch_count` on any
  module that uses `ScheduledFloat`; this is crucial to having all scheduled
  hyperparameters in sync.
- `get_parameter_groups_with_lrs` – computes parameter groups with
  per-module LR scaling based on an attribute `lr_scale` that can be set
  anywhere in the module hierarchy.

### 8.3. Inference utilities

**File:** `zipvoice/utils/infer.py`

- `chunk_tokens_punctuation` – splits text into sub-sequences at punctuation,
  limiting chunk size.
- `chunk_tokens_dialog` – splits at `[S1]` tags.
- `batchify_tokens` – groups chunks into mini-batches based on a target
  duration budget (estimated from token duration and prompt duration).
- `cross_fade_concat` – concatenates multiple audio chunks with overlap-add
  cross-fading to avoid boundary artifacts.
- `remove_silence`, `remove_silence_edges` – use `pydub` for silence
  detection and trimming.

These utilities feed into CLI inference scripts for multi-sentence or dialog
synthesis, not into the LuxTTS Python class directly.

---

## 9. Putting it all together – what LuxTTS actually is

From an architectural perspective, LuxTTS is:

- A **ZipVoiceDistill** flow-matching acoustic model with:
  - Zipformer backbone.
  - Velocity parameterization `v = x1 − x0` over straight-line flows between
    noise and data.
  - Deterministic, anchor-based sampler (no noise injection after t=0).
  - Classifier-free guidance that handles **text** and **speech prompt**
    differently in low- vs high-t regimes.
- A **prompt conditioning pipeline** that:
  - Uses Whisper ASR to recover the prompt transcript.
  - Uses the prompt’s mel features as speech condition.
  - Uses an EmiliaTokenizer-based phone sequence for both prompt and target
    text conditions.
- A **24 kHz → 48 kHz vocoder stack** that allows configurable bandwidth and
  output sample rate via `linacodec.vocoder.vocos.Vocos`.
- A set of **deployment backends**:
  - Pure PyTorch (ZipVoiceDistill) for GPU/MPS.
  - ONNX Runtime (`OnnxModel`) for CPU.
  - TensorRT (`TrtContextWrapper`) for fast CUDA inference of the fm_decoder.

For development, the critical semantic invariants to keep in mind are:

1. The acoustic model is trained on **continuous-time, straight-line
   flow-matching**, with a **velocity parameterization**. Any change to
   sampling that assumes an ODE/SDE discretization beyond that will change
   semantics.
2. Classifier-free guidance behavior is **asymmetric in t** and uses speech
   prompts differently at low and high t; simplifying this changes
   conditioning semantics.
3. All architectural tricks in Zipformer (Balancers, Whiteners, ScheduledFloat,
   Swoosh activations) are there to enable very aggressive training while
   keeping gradients stable; turning them off or modifying schedules will
   affect trainability more than inference.
4. LuxTTS’s **48 kHz output** is purely a vocoder concern; the acoustic model
   remains at 24 kHz mel resolution. Changing feature configs or sample
   rates means touching `VocosFbank`, the model configs, and the vocoder
   front/back.

With those invariants respected, the system is modular enough that you can:

- Swap in new tokenizers (e.g., language-specific phonemizers) by adhering
  to the token file and shape conventions.
- Experiment with new time schedules or solver variants by reusing the
  `EulerSolver` interface.
- Replace or augment the vocoder while leaving the acoustic model untouched.
- Extend to new conditioning modalities (e.g., emotion tags) by expanding
  `xt` concatenation and adjusting Zipformer input dimensions.
