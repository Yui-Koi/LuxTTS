# Flow-Matching ODE Solver in LuxTTS / ZipVoice

This document details the flow-matching ODE solver used during inference and distillation in LuxTTS/ZipVoice. It explains how the solver is implemented, how it interacts with the models, and which hyperparameters most strongly affect performance and quality.

Relevant files:

- `zipvoice/models/modules/solver.py`
- `zipvoice/models/zipvoice.py`
- `zipvoice/models/zipvoice_distill.py`
- `zipvoice/onnx_modeling.py`
- `zipvoice/utils/tensorrt.py`

---

## 1. Flow Matching Formulation

Training in `ZipVoice.forward` (in `zipvoice/models/zipvoice.py`) uses a standard flow-matching formulation between pure noise `x_0` and clean data `x_1`.

Given:

- `features`  ≈ `x_1`  (clean acoustic features)
- `noise`     ≈ `x_0`  (sampled from N(0, I))
- `t`         ∈ (0, 1), broadcast to `(B, 1, 1)`

The training pipeline constructs:

```python
xt = features * t + noise * (1 - t)  # (B, T, F)
ut = features - noise               # (B, T, F)
```

The decoder predicts a **velocity** field `v(x, t)`:

```python
vt = self.forward_fm_decoder(
    t=t,
    xt=xt,
    text_condition=text_condition,
    speech_condition=speech_condition,
    padding_mask=padding_mask,
)

fm_loss = torch.mean((vt[loss_mask] - ut[loss_mask]) ** 2)
```

Conceptually, the data path is:

\[
  x_t = (1 - t) x_0 + t x_1, \quad v = x_1 - x_0.
\]

At training time, the model learns to map `(x_t, t, conditions)` to the **target velocity** `v = x_1 - x_0` under various masks, conditioning drops, and prompt configurations.

---

## 2. Runtime Solver Stack

### 2.1 DiffusionModel (classifier-free guidance wrapper)

Defined in `zipvoice/models/modules/solver.py`:

```python
class DiffusionModel(torch.nn.Module):
    def __init__(self, model, func_name="forward_fm_decoder"):
        self.model = model
        self.func_name = func_name
        self.model_func = getattr(self.model, func_name)

    def forward(
        self,
        t, x,
        text_condition,
        speech_condition,
        padding_mask=None,
        guidance_scale=0.0,
        **kwargs,
    ):
        ...
```

Responsibilities:

- Wraps a model method (`ZipVoice.forward_fm_decoder` or dialog equivalents).
- Implements **classifier-free guidance (CFG)** at inference time.

CFG behavior (simplified):

- If `guidance_scale == 0`, run **single** forward:
  ```python
  return self.model_func(...)
  ```
- Else, assume `t` is scalar (`t.dim() == 0`) and do:
  ```python
  x = torch.cat([x] * 2, dim=0)            # (2B, T, F)
  padding_mask = torch.cat([padding_mask] * 2, dim=0)

  # Unconditional branch has zero text_condition
  text_condition = torch.cat(
      [torch.zeros_like(text_condition), text_condition],
      dim=0,
  )

  if t > 0.5:
      # Late timesteps: uncond has no prompt, cond has prompt
      speech_condition = torch.cat(
          [torch.zeros_like(speech_condition), speech_condition],
          dim=0,
      )
  else:
      # Early timesteps: both branches see prompt, strengthen text guidance
      guidance_scale = guidance_scale * 2
      speech_condition = torch.cat(
          [speech_condition, speech_condition],
          dim=0,
      )

  data_uncond, data_cond = self.model_func(...).chunk(2, dim=0)
  res = (1 + guidance_scale) * data_cond - guidance_scale * data_uncond
  ```

Key points:

- **Text CFG**: unconditional branch always zeros text_condition.
- **Prompt CFG**:
  - For **early** timesteps (`t <= 0.5`): both branches see the same prompt; CFG only contrasts text vs no-text while preserving timbre.
  - For **late** timesteps (`t > 0.5`): uncond branch sees no prompt, cond branch keeps the prompt; CFG now modulates prompt influence too.
- Runtime cost: **2 forward passes** through the decoder per ODE step when `guidance_scale != 0`.

This wrapper is used for the original (non-distilled) ZipVoice model.

### 2.2 DistillDiffusionModel

`DistillDiffusionModel` inherits `DiffusionModel` but removes the CFG batching logic and directly forwards `guidance_scale` into the model:

```python
class DistillDiffusionModel(DiffusionModel):
    def forward(..., guidance_scale=0.0, **kwargs):
        if not torch.is_tensor(guidance_scale):
            guidance_scale = torch.tensor(...)
        return self.model_func(
            t=t,
            xt=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            guidance_scale=guidance_scale,
            **kwargs,
        )
```

This is used by `ZipVoiceDistill` and assumes the decoder (`TTSZipformer`) has been trained with **guidance-scale embeddings** baked into it.

Result: **one forward pass** per ODE step.

### 2.3 EulerSolver

Defined in `zipvoice/models/modules/solver.py`:

```python
class EulerSolver:
    def __init__(self, model, func_name="forward_fm_decoder"):
        self.model = DiffusionModel(model, func_name=func_name)

    def sample(
        self,
        x,
        text_condition,
        speech_condition,
        padding_mask,
        num_step=10,
        guidance_scale=0.0,
        t_start=0.0,
        t_end=1.0,
        t_shift=1.0,
        **kwargs,
    ):
        timesteps = get_time_steps(...)
        for step in range(num_step):
            t_cur  = timesteps[step]
            t_next = timesteps[step + 1]

            v = self.model(
                t=t_cur,
                x=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                **kwargs,
            )

            x_1_pred = x + (1.0 - t_cur) * v
            x_0_pred = x - t_cur * v

            if step < num_step - 1:
                x = (1.0 - t_next) * x_0_pred + t_next * x_1_pred
            else:
                x = x_1_pred
        return x
```

Interpretation:

- At each step we treat `x` as `x_t` at time `t_cur`.
- Given `v = x_1 - x_0`, we can solve for `(x_0, x_1)` from `(x_t, v, t)` using the linear relation:
  - `x_1_pred = x_t + (1 - t) v`
  - `x_0_pred = x_t - t v`
- The **next** state is taken as the exact linear interpolation at `t_next`:
  \[
    x_{t_{\text{next}}} \approx (1 - t_{\text{next}}) x_0^{\text{pred}} + t_{\text{next}} x_1^{\text{pred}}.
  \]
- On the final step, we "snap" directly to `x_1_pred` as the clean data estimate.

This uses the flow-matching structure (the line between noise and data) to define an **anchor-based Euler** update that is more stable than naive Euler integration.

### 2.4 DistillEulerSolver

```python
class DistillEulerSolver(EulerSolver):
    def __init__(self, model, func_name="forward_fm_decoder"):
        self.model = DistillDiffusionModel(model, func_name=func_name)
```

The algorithm is identical to `EulerSolver`, but uses `DistillDiffusionModel` so that:

- Only **one** decoder call per step.
- `guidance_scale` is interpreted via a guidance-scale embedding inside `TTSZipformer`.

### 2.5 Time Steps and `t_shift`

`get_time_steps` generates a monotonically increasing schedule:

```python
def get_time_steps(t_start=0.0, t_end=1.0, num_step=10, t_shift=1.0, device=...):
    timesteps = torch.linspace(t_start, t_end, num_step + 1, device=device)
    if t_shift == 1.0:
        return timesteps

    inv_s = 1.0 / t_shift
    denom = torch.add(inv_s, timesteps, alpha=1.0 - inv_s)
    return timesteps.div_(denom)
```

This applies a rational time-warp:

\[
  t' = \frac{t}{1 / t_{\text{shift}} + (1 - 1 / t_{\text{shift}}) t}.
\]

- With `t_shift < 1` (recommended range `(0, 1]`):
  - Early timesteps are **compressed**; the solver spends more resolution in the low-SNR, noisy region.
- With `t_shift = 1`:
  - Linear schedule.

Together, `num_step` and `t_shift` control **where** the solver spends its computational budget along the trajectory.

---

## 3. Integration with ZipVoice and ZipVoiceDistill

### 3.1 ZipVoice.sample (full generation)

In `zipvoice/models/zipvoice.py`:

```python
self.solver = EulerSolver(self, func_name="forward_fm_decoder")
...

def sample(
    self,
    tokens,
    prompt_tokens,
    prompt_features,
    prompt_features_lens,
    features_lens=None,
    speed: float = 1.0,
    t_shift: float = 1.0,
    duration: str = "predict",
    num_step: int = 5,
    guidance_scale: float = 0.5,
):
    ...
```

Key steps:

1. **Text conditioning and length prediction**
   - If `duration == "predict"`, frame lengths are predicted from token counts and `speed` using `forward_text_inference_ratio_duration`.
   - Otherwise, uses ground-truth `features_lens`.
   - Output: `text_condition` (`B, T, F`) and `padding_mask` (`B, T`).

2. **Speech condition from prompt**
   - Prompt features are padded to match `T` and masked so that only prompt frames carry non-zero values:
     ```python
     speech_condition = F.pad(prompt_features, (0, 0, 0, num_frames - prompt_features.size(1)))
     speech_condition_mask = make_pad_mask(prompt_features_lens, num_frames)
     speech_condition = torch.where(
         speech_condition_mask.unsqueeze(-1),
         torch.zeros_like(speech_condition),
         speech_condition,
     )
     ```

3. **Initialization**
   - Sample `x_0 ~ N(0, I)` of shape `(B, T, F)`.

4. **ODE sampling**
   - Call the solver:
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

5. **Splitting prompt vs generated region**
   - `x1` includes both prompt and generated frames.
   - The code computes lengths and slices out:
     - `x1_prompt`: reconstructed prompt region.
     - `x1_wo_prompt`: generated continuation.

### 3.2 ZipVoice.sample_intermediate (distillation helper)

```python
def sample_intermediate(
    self,
    tokens,
    features,
    features_lens,
    noise,
    speech_condition_mask,
    t_start: float,
    t_end: float,
    num_step: int = 1,
    guidance_scale: torch.Tensor = None,
):
    text_condition, padding_mask = self.forward_text_train(tokens, features_lens)
    speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, features)

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
    x_t_end_lens = (~padding_mask).sum(-1)
    return x_t_end, x_t_end_lens
```

This is used heavily during **distillation**, where the student model is trained to reproduce the teacher’s behavior over intermediate time intervals `(t_start, t_end)`.

### 3.3 ZipVoiceDistill

`ZipVoiceDistill` (in `zipvoice/models/zipvoice_distill.py`) swaps in a distilled decoder and solver:

```python
self.fm_decoder = TTSZipformer(..., use_guidance_scale_embed=True)
self.solver = DistillEulerSolver(self, func_name="forward_fm_decoder")

# During training, forward is just an ODE sample between timesteps

def forward(...):
    return self.sample_intermediate(...)
```

Notes:

- The distilled decoder takes `guidance_scale` as an **additional embedding** (via `use_guidance_scale_embed=True`).
- This removes the need for runtime CFG logic and the extra decoder pass per step.

---

## 4. ONNX / CPU Solver Path

The ONNX CPU path in `zipvoice/onnx_modeling.py` reimplements the same ODE logic but drives ONNXRuntime sessions instead of PyTorch modules.

```python
from zipvoice.models.modules.solver import get_time_steps

class OnnxModel:
    ...
    def run_fm_decoder(...):
        out = self.fm_decoder.run(...)
        return torch.from_numpy(out[0])


def sample(
    model: OnnxModel,
    tokens,
    prompt_tokens,
    prompt_features,
    speed: float = 1.3,
    t_shift: float = 0.5,
    guidance_scale: float = 1.0,
    num_step: int = 16,
):
    ...
    text_condition = model.run_text_encoder(...)
    timesteps = get_time_steps(t_start=0.0, t_end=1.0, num_step=num_step, t_shift=t_shift)

    x = torch.randn(batch_size, num_frames, feat_dim)
    speech_condition = F.pad(prompt_features, ...)
    guidance_scale = torch.tensor(guidance_scale, dtype=torch.float32)

    for step in range(num_step):
        t_cur  = timesteps[step]
        t_next = timesteps[step + 1]
        v = model.run_fm_decoder(
            t=t_cur,
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale,
        )

        x_1_pred = x + (1.0 - t_cur) * v
        x_0_pred = x - t_cur * v

        if step < num_step - 1:
            x = (1.0 - t_next) * x_0_pred + t_next * x_1_pred
        else:
            x = x_1_pred

    x = x[:, prompt_features_len.item():, :]
    return x
```

Key points:

- Exactly the same ODE update rule as `EulerSolver.sample`.
- No Python-side CFG logic; **guidance_scale** is passed directly to the ONNX decoder, which is a distilled-style network.
- The Python loop is still over `num_step`, but each step is a relatively cheap ONNX forward compared to GPU.

Note: Similar logic is duplicated in `zipvoice/bin/infer_zipvoice_onnx.py`. Any solver algorithm change must be applied consistently to both.

---

## 5. TensorRT Interaction

`zipvoice/utils/tensorrt.py` provides `load_trt`, which replaces `model.fm_decoder` with a TensorRT-backed wrapper:

```python
def load_trt(model: nn.Module, trt_model: str, trt_concurrent: int = 1):
    import tensorrt as trt
    ...
    del model.fm_decoder
    model.fm_decoder = TrtContextWrapper(estimator_engine, ...)
```

`TrtContextWrapper.__call__`:

```python
def __call__(self, x, t, padding_mask, guidance_scale=None):
    x = x.to(torch.float16)
    t = t.to(torch.float16)
    padding_mask = padding_mask.to(torch.float16)
    if guidance_scale is not None:
        guidance_scale = guidance_scale.to(torch.float16)

    [estimator, stream], trt_engine = self.acquire_estimator()
    batch_size, seq_len = x.size(0), x.size(1)

    output = torch.empty(batch_size, seq_len, self.feat_dim, dtype=x.dtype, device=x.device)

    with stream:
        estimator.set_input_shape('x', (batch_size, x.size(1), x.size(2)))
        estimator.set_input_shape('t', (batch_size,))
        estimator.set_input_shape('padding_mask', (batch_size, padding_mask.size(1)))
        if guidance_scale is not None:
            estimator.set_input_shape('guidance_scale', (batch_size,))
        ...
        estimator.execute_async_v3(...)
    return output.to(torch.float32)
```

Implications:

- All **ODE logic remains in Python** (`EulerSolver.sample` or `DistillEulerSolver.sample`).
- Only the decoder network itself is accelerated with TensorRT.
- Any changes to the decoder inputs/outputs or to guidance handling require regenerating the TensorRT engine and updating this wrapper.

---

## 6. Important Hyperparameters and Their Effects

### 6.1 `num_step`

- Number of ODE integration steps.
- Direct linear multiplier for runtime.
- Typical defaults:
  - Original ZipVoice: `num_step ≈ 16`.
  - ZipVoiceDistill: `num_step ≈ 8`.

### 6.2 `t_shift`

- Rational time warp applied to `linspace(t_start, t_end, num_step + 1)`.
- `t_shift < 1` concentrates steps near the **low-SNR** (noisier) part of the trajectory.
- Tuning `t_shift` with smaller `num_step` can often preserve quality while reducing cost.

### 6.3 `guidance_scale`

- Non-distilled (PyTorch):
  - Controls classifier-free guidance contrast between conditional and unconditional branches.
  - `guidance_scale > 0` roughly means "follow the conditional branch more strongly".
  - Increases runtime by ~2× due to dual forward passes per step.
- Distilled / ONNX / TensorRT:
  - Interpreted by the decoder as an embedding, with no extra forward passes.
  - Quality/runtime trade-off is mostly architectural/training-time, not runtime.

### 6.4 `speed`

- Used in duration prediction to estimate `T` (number of frames):
  \[
    T \approx T_{\text{prompt}} + \left\lceil \frac{T_{\text{prompt}}}{L_{\text{prompt tokens}}} \cdot \frac{L_{\text{text tokens}}}{\text{speed}} \right\rceil.
  \]
- Smaller `speed` → longer predicted duration → more frames → higher ODE cost.
- ODE logic itself is agnostic to `speed`; it just runs over the final `T`.

### 6.5 `feat_scale`

- Features are scaled by `feat_scale` (default `0.1`) before entering the model.
- Both training and inference operate in this scaled space; outputs are rescaled (`/ feat_scale`) before the vocoder.
- Changing this without retraining affects the dynamics of the ODE and is not recommended.

---

## 7. Complexity and Performance Picture

Let:

- `B` = batch size
- `T` = number of frames per sequence
- `F` = feature dimension (100 mono, 200 stereo)
- `S` = `num_step`

Then per call to `ZipVoice.sample` on GPU:

- Non-distilled, with CFG:
  - Cost ~ `O(S * 2 * C_decoder(B, T, 3F))`.
- Distilled, with guidance embedding:
  - Cost ~ `O(S * 1 * C_decoder(B, T, 3F))`.

The additional per-step algebra (`x_1_pred`, `x_0_pred` updates) is `O(B * T * F)` and usually negligible compared to decoder cost.

In practice, the most impactful levers for speed vs. quality are:

1. **Choosing the distilled model** where possible.
2. **Reducing `num_step`** while adjusting `t_shift` to keep perceptual quality.
3. **Controlling sequence length `T`** via `speed` and text chunking.

---

## 8. Practical Optimization Directions

When optimizing the ODE solver path, the following strategies are most promising:

1. **Prefer ZipVoiceDistill over ZipVoice for deployment**
   - Removes runtime CFG overhead.
   - Trained to approximate CFG behavior with fewer steps.

2. **Tune `num_step` and `t_shift` jointly**
   - Start from a higher-quality configuration (e.g., `num_step = 16, t_shift = 0.5`), then
     gradually reduce `num_step` while adjusting `t_shift` to emphasize low-SNR regions.

3. **Keep ONNX and PyTorch solver logic aligned**
   - Any algorithmic change (e.g., higher-order integrator) must be mirrored in both
     `EulerSolver.sample` and the ONNX `sample()` implementation.

4. **Be aware of TensorRT constraints**
   - Changing decoder input signatures or guidance handling requires regenerating the TRT
     engine and updating `TrtContextWrapper`.

For a high-level architectural summary of where the solver fits into the whole system, see the **Core Modeling Components** and **Inference Pipelines** sections in `architecture.md`.
