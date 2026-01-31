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

from typing import Optional, Union, Tuple

import torch


def flow_matching_cfg_factors(
    t: torch.Tensor, guidance_scale: torch.Tensor, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the late flag and effective guidance scale for flow-matching CFG.

    Args:
        t:
            Scalar tensor with current time in (0, 1).
        guidance_scale:
            Guidance scale tensor (broadcastable) used for CFG.
        dtype:
            Target dtype for the returned `late` tensor.

    Returns:
        late:
            Tensor with value 0.0 if t <= 0.5, 1.0 if t > 0.5 (in `dtype`).
        s_eff:
            Effective guidance scale tensor, where early timesteps use
            ``2 * guidance_scale`` and late timesteps use ``1 * guidance_scale``.
    """
    late = (t > 0.5).to(dtype)
    s_eff = guidance_scale * (2.0 - late)
    return late, s_eff


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

        # Scratch buffers for CFG to reduce per-step allocations.
        self._x2: Optional[torch.Tensor] = None
        self._speech2: Optional[torch.Tensor] = None
        self._text2: Optional[torch.Tensor] = None
        self._pad2: Optional[torch.Tensor] = None

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """Forward function that handles classifier-free guidance.

        Args:
            t:
                Current timestep, a tensor containing a single scalar in (0, 1).
            x:
                Current state `x_t`, shape (batch, seq_len, emb_dim).
            text_condition:
                Text condition embeddings, shape (batch, seq_len, emb_dim).
            speech_condition:
                Speech condition (prompt) embeddings, shape (batch, seq_len, emb_dim).
            padding_mask:
                Padding mask; True means masked position, shape (batch, seq_len).
            guidance_scale:
                Classifier-free guidance scale. Either a Python float or a tensor of
                shape (batch, 1, 1) / scalar tensor.

        Returns:
            Predicted velocity with shape (batch, seq_len, emb_dim).
        """
        # Fast path: scalar guidance_scale == 0.0 -> no CFG, single forward pass.
        if not torch.is_tensor(guidance_scale):
            gs = float(guidance_scale)
            if gs == 0.0:
                return self.model_func(
                    t=t,
                    xt=x,
                    text_condition=text_condition,
                    speech_condition=speech_condition,
                    padding_mask=padding_mask,
                    **kwargs,
                )
            guidance_scale = torch.tensor(gs, dtype=t.dtype, device=t.device)

        # At this point guidance_scale is a tensor; we always apply CFG.
        # We avoid converting tensors to Python scalars to prevent GPU syncs.
        assert t.dim() == 0, "t is expected to be a scalar tensor for CFG"

        # Shapes and dtypes
        batch_size, seq_len, feat_dim = x.shape
        device = x.device

        # Allocate or reuse 2x-batch scratch buffers.
        if (
            self._x2 is None
            or self._x2.shape != (2 * batch_size, seq_len, feat_dim)
            or self._x2.device != device
            or self._x2.dtype != x.dtype
        ):
            self._x2 = x.new_empty(2 * batch_size, seq_len, feat_dim)

        if (
            self._speech2 is None
            or self._speech2.shape != speech_condition.shape[:2] + (feat_dim,)
            or self._speech2.device != speech_condition.device
            or self._speech2.dtype != speech_condition.dtype
        ):
            self._speech2 = speech_condition.new_empty(2 * batch_size, seq_len, feat_dim)

        if padding_mask is not None:
            if (
                self._pad2 is None
                or self._pad2.shape != (2 * batch_size, seq_len)
                or self._pad2.device != padding_mask.device
                or self._pad2.dtype != padding_mask.dtype
            ):
                self._pad2 = padding_mask.new_empty(2 * batch_size, seq_len)
            pad2 = self._pad2
        else:
            pad2 = None

        if (
            self._text2 is None
            or self._text2.shape != (2 * batch_size, seq_len, feat_dim)
            or self._text2.device != text_condition.device
            or self._text2.dtype != text_condition.dtype
        ):
            self._text2 = text_condition.new_empty(2 * batch_size, seq_len, feat_dim)
        text2 = self._text2

        # Fill x2 = [x; x]
        self._x2[:batch_size].copy_(x)
        self._x2[batch_size:].copy_(x)

        # Fill pad2 = [pad; pad] if provided
        if pad2 is not None and padding_mask is not None:
            pad2[:batch_size].copy_(padding_mask)
            pad2[batch_size:].copy_(padding_mask)

        # Text condition: [zeros; text]
        text2[:batch_size].zero_()
        text2[batch_size:].copy_(text_condition)

        # Compute late flag and effective guidance scale on device
        late, s_eff = flow_matching_cfg_factors(
            t=t, guidance_scale=guidance_scale, dtype=speech_condition.dtype
        )

        # Speech condition: uncond = speech * (1 - late), cond = speech
        speech2 = self._speech2
        speech2[:batch_size].copy_(speech_condition)
        speech2[:batch_size].mul_(1.0 - late)
        speech2[batch_size:].copy_(speech_condition)

        data_uncond, data_cond = self.model_func(
            t=t,
            xt=self._x2,
            text_condition=text2,
            speech_condition=speech2,
            padding_mask=pad2,
            **kwargs,
        ).chunk(2, dim=0)

        # v = cond + s_eff * (cond - uncond)
        res = data_cond + s_eff * (data_cond - data_uncond)
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

        if num_step <= 0:
            return x

        # Pre-compute time intervals on device: dt_i = t_{i+1} - t_i
        dts = timesteps[1:] - timesteps[:-1]

        # First num_step - 1 steps: linear update x_{t+dt} = x_t + dt * v
        for step in range(max(num_step - 1, 0)):
            t_cur = timesteps[step]
            dt = dts[step]

            v = self.model(
                t=t_cur,
                x=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                **kwargs,
            )

            # In-place update to avoid extra allocations
            x.add_(v, alpha=dt)

        # Final step: snap to the predicted clean data x_1
        t_cur = timesteps[num_step - 1]
        v = self.model(
            t=t_cur,
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            guidance_scale=guidance_scale,
            **kwargs,
        )
        x.add_(v, alpha=(1.0 - t_cur))

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

    timesteps = torch.linspace(t_start, t_end, num_step + 1, device=device)

    if t_shift == 1.0:
        return timesteps
    
    inv_s = 1.0 / t_shift
    denom = torch.add(inv_s, timesteps, alpha=1.0 - inv_s)
    return timesteps.div_(denom)
