#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import torch


class DiffusionModel(torch.nn.Module):
    """A wrapper of diffusion models for inference.
    Args:
        model: The diffusion model.
        func_name: The function name to call.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        func_name: str = "forward_fm_decoder",
    ):
        super().__init__()
        self.model = model
        self.func_name = func_name
        self.model_func = getattr(self.model, func_name)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        # Explicitly list CFG params to filter them from kwargs
        use_cfg: bool = False,
        speech_gate_open: bool = False,
        text_condition_cfg: Optional[torch.Tensor] = None,
        speech_condition_cfg_early: Optional[torch.Tensor] = None,
        speech_condition_cfg_late: Optional[torch.Tensor] = None,
        padding_mask_cfg: Optional[torch.Tensor] = None,
        x_buffer: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward function that Handles the classifier-free guidance.
        Args:
            t: The current timestep, a tensor of a tensor of a single float.
            x: The initial value, with the shape (batch, seq_len, emb_dim).
            text_condition: The text_condition of the diffision model, with
                the shape (batch, seq_len, emb_dim).
            speech_condition: The speech_condition of the diffision model, with the
                shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding; True means masked position, with the
                shape (batch, seq_len).
            guidance_scale: The scale of classifier-free guidance, a float or a tensor
                of shape (batch, 1, 1).
        Retrun:
            The prediction with the shape (batch, seq_len, emb_dim).
        """
        if not torch.is_tensor(guidance_scale):
            guidance_scale = torch.tensor(
                guidance_scale, dtype=t.dtype, device=t.device
            )

        # Check use_cfg flag first (avoiding tensor sync if possible)
        use_cfg = kwargs.get("use_cfg", None)
        if use_cfg is None:
            # Fallback for backward compatibility or if called directly without flags
            use_cfg = not (guidance_scale == 0.0).all()

        if not use_cfg:
            if t.dim() == 0:
                t = t.expand(x.shape[0])
            return self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                **kwargs
            )
        else:
            assert t.dim() == 0

            # Optimizations: reuse preallocated buffers if available
            x_buffer = kwargs.get("x_buffer", None)
            text_condition_cfg = kwargs.get("text_condition_cfg", None)
            speech_condition_cfg_early = kwargs.get("speech_condition_cfg_early", None)
            speech_condition_cfg_late = kwargs.get("speech_condition_cfg_late", None)
            padding_mask_cfg = kwargs.get("padding_mask_cfg", None)

            if x_buffer is not None:
                B = x.shape[0]
                x_buffer[:B].copy_(x)
                x_buffer[B:].copy_(x)
                x_in = x_buffer
            else:
                x_in = torch.cat([x] * 2, dim=0)

            if padding_mask_cfg is not None:
                padding_mask_in = padding_mask_cfg
            else:
                padding_mask_in = torch.cat([padding_mask] * 2, dim=0) if padding_mask is not None else None

            if text_condition_cfg is not None:
                text_condition_in = text_condition_cfg
            else:
                text_condition_in = torch.cat(
                    [torch.zeros_like(text_condition), text_condition], dim=0
                )

            # Use precomputed speech_gate_open if available, else fallback to t > 0.5 (sync!)
            speech_gate_open = kwargs.get("speech_gate_open", None)
            if speech_gate_open is None:
                speech_gate_open = (t > 0.5).item()

            if speech_gate_open:
                if speech_condition_cfg_late is not None:
                    speech_condition_in = speech_condition_cfg_late
                else:
                    speech_condition_in = torch.cat(
                        [torch.zeros_like(speech_condition), speech_condition], dim=0
                    )
            else:
                guidance_scale = guidance_scale * 2
                if speech_condition_cfg_early is not None:
                    speech_condition_in = speech_condition_cfg_early
                else:
                    speech_condition_in = torch.cat(
                        [speech_condition, speech_condition], dim=0
                    )

            if t.dim() == 0:
                t = t.expand(x_in.shape[0])

            data_uncond, data_cond = self.model_func(
                t=t,
                xt=x_in,
                text_condition=text_condition_in,
                speech_condition=speech_condition_in,
                padding_mask=padding_mask_in,
                **kwargs
            ).chunk(2, dim=0)

            res = (1 + guidance_scale) * data_cond - guidance_scale * data_uncond
            return res


class DistillDiffusionModel(DiffusionModel):
    """A wrapper of distilled diffusion models for inference.
    Args:
        model: The distilled diffusion model.
        func_name: The function name to call.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        func_name: str = "forward_fm_decoder",
    ):
        super().__init__(model=model, func_name=func_name)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        # Explicitly list CFG params to filter them from kwargs passed to model_func
        use_cfg: bool = False,
        speech_gate_open: bool = False,
        text_condition_cfg: Optional[torch.Tensor] = None,
        speech_condition_cfg_early: Optional[torch.Tensor] = None,
        speech_condition_cfg_late: Optional[torch.Tensor] = None,
        padding_mask_cfg: Optional[torch.Tensor] = None,
        x_buffer: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward function that Handles the classifier-free guidance.
        Args:
            t: The current timestep, a tensor of a single float.
            x: The initial value, with the shape (batch, seq_len, emb_dim).
            text_condition: The text_condition of the diffision model, with
                the shape (batch, seq_len, emb_dim).
            speech_condition: The speech_condition of the diffision model, with the
                shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding; True means masked position, with the
                shape (batch, seq_len).
            guidance_scale: The scale of classifier-free guidance, a float or a tensor
                of shape (batch, 1, 1).
        Retrun:
            The prediction with the shape (batch, seq_len, emb_dim).
        """
        if not torch.is_tensor(guidance_scale):
            guidance_scale = torch.tensor(
                guidance_scale, dtype=t.dtype, device=t.device
            )
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        return self.model_func(
            t=t,
            xt=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            guidance_scale=guidance_scale,
            **kwargs
        )


class EulerSolver:
    def __init__(
        self,
        model: torch.nn.Module,
        func_name: str = "forward_fm_decoder",
    ):
        """Construct a Euler Solver
        Args:
            model: The diffusion model.
            func_name: The function name to call.
        """
        self.model = DiffusionModel(model, func_name=func_name)

    def sample(
        self,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: torch.Tensor,
        num_step: int = 10,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        t_start: float = 0.0,
        t_end: float = 1.0,
        t_shift: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        device = x.device
        assert isinstance(t_start, float) and isinstance(t_end, float)

        # Generate the schedule of timesteps
        timesteps = get_time_steps(
            t_start=t_start,
            t_end=t_end,
            num_step=num_step,
            t_shift=t_shift,
            device=device,
        )

        # Precompute dt
        dt = timesteps[1:] - timesteps[:-1]

        # Determine CFG usage once
        if isinstance(guidance_scale, float):
            use_cfg = (guidance_scale != 0.0)
        else:
            # Assume tensor guidance implies CFG usage
            use_cfg = True

        # Precompute speech gating (t > 0.5) to avoid sync inside loop
        # timesteps is on device, move to cpu for bool check
        timesteps_cpu = timesteps.detach().cpu()
        speech_gating = (timesteps_cpu > 0.5).tolist()

        # Workstream 3: Eliminate per-step allocations in CFG wrapper
        text_condition_cfg = None
        speech_condition_cfg_early = None
        speech_condition_cfg_late = None
        padding_mask_cfg = None
        x_buffer = None

        if use_cfg:
            # Precompute concatenated conditions once
            text_condition_cfg = torch.cat(
                [torch.zeros_like(text_condition), text_condition], dim=0
            )

            speech_condition_cfg_late = torch.cat(
                [torch.zeros_like(speech_condition), speech_condition], dim=0
            )
            speech_condition_cfg_early = torch.cat(
                [speech_condition, speech_condition], dim=0
            )

            if padding_mask is not None:
                padding_mask_cfg = torch.cat([padding_mask, padding_mask], dim=0)

            # Preallocate buffer for x doubling (Strategy A)
            B_sz = x.shape[0]
            x_buffer = torch.empty(
                (2 * B_sz, *x.shape[1:]), dtype=x.dtype, device=x.device
            )

        for step in range(num_step):
            t_cur = timesteps[step]

            # Predict velocity (v)
            v = self.model(
                t=t_cur,
                x=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                use_cfg=use_cfg,
                speech_gate_open=speech_gating[step],
                text_condition_cfg=text_condition_cfg,
                speech_condition_cfg_early=speech_condition_cfg_early,
                speech_condition_cfg_late=speech_condition_cfg_late,
                padding_mask_cfg=padding_mask_cfg,
                x_buffer=x_buffer,
                **kwargs
            )

            # Euler update: x = x + dt * v
            # Note: if t_end=1.0, this is mathematically equivalent to the previous
            # 'snap to x_pred' at the last step.
            x.add_(v, alpha=dt[step])

        return x


class DistillEulerSolver(EulerSolver):
    def __init__(
        self,
        model: torch.nn.Module,
        func_name: str = "forward_fm_decoder",
    ):
        """Construct a Euler Solver for distilled diffusion models.
        Args:
            model: The diffusion model.
        """
        self.model = DistillDiffusionModel(model, func_name=func_name)


def get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute the intermediate time steps for diffusion sampling.
    Applies monotonic shift: t' = t_shift * t / (1 + (t_shift - 1) * t)
    Equivalently: t' = t / (inv_s + alpha * t), where inv_s = 1/t_shift, alpha = 1 - inv_s
       
    Args:
        t_start: The starting time of the sampling (default is 0).
        t_end: The starting time of the sampling (default is 1).
        num_step: The number of sampling.
        t_shift: shift the t toward smaller numbers so that the sampling
            will emphasize low SNR region. Should be in the range of (0, 1].
            The shifting will be more significant when the number is smaller.
        device: A torch device.
    Returns:
        The time step with the shape (num_step + 1,).
    """

    timesteps = torch.linspace(t_start, t_end, num_step + 1, device=device, dtype=torch.float32)

    if t_shift == 1.0:
        return timesteps
    
    inv_s = 1.0 / t_shift
    denom = torch.add(inv_s, timesteps, alpha=1.0 - inv_s)
    return timesteps.div_(denom)
