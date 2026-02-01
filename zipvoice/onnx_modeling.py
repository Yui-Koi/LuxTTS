import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from torch import Tensor, nn

from zipvoice.bin.infer_zipvoice import get_vocoder
from zipvoice.models.modules.solver import get_time_steps
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.common import AttributeDict, str2bool
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    chunk_tokens_punctuation,
    cross_fade_concat,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)

class OnnxModel:
    def __init__(
        self,
        text_encoder_path: str,
        fm_decoder_path: str,
        num_thread: int = 1,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = num_thread
        session_opts.intra_op_num_threads = num_thread

        self.session_opts = session_opts

        self.init_text_encoder(text_encoder_path)
        self.init_fm_decoder(fm_decoder_path)

    def init_text_encoder(self, model_path: str):
        self.text_encoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_fm_decoder(self, model_path: str):
        self.fm_decoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])

    def run_text_encoder(
        self,
        tokens: Tensor,
        prompt_tokens: Tensor,
        prompt_features_len: Tensor,
        speed: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        out = self.text_encoder.run(
            [
                self.text_encoder.get_outputs()[0].name,
            ],
            {
                self.text_encoder.get_inputs()[0].name: tokens.numpy(),
                self.text_encoder.get_inputs()[1].name: prompt_tokens.numpy(),
                self.text_encoder.get_inputs()[2].name: prompt_features_len.numpy(),
                self.text_encoder.get_inputs()[3].name: speed.numpy(),
            },
        )
        return torch.from_numpy(out[0])

    def run_fm_decoder(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: torch.Tensor,
        guidance_scale: Tensor,
    ) -> Tensor:
        out = self.fm_decoder.run(
            [
                self.fm_decoder.get_outputs()[0].name,
            ],
            {
                self.fm_decoder.get_inputs()[0].name: t.numpy(),
                self.fm_decoder.get_inputs()[1].name: x.numpy(),
                self.fm_decoder.get_inputs()[2].name: text_condition.numpy(),
                self.fm_decoder.get_inputs()[3].name: speech_condition.numpy(),
                self.fm_decoder.get_inputs()[4].name: guidance_scale.numpy(),
            },
        )
        return torch.from_numpy(out[0])

def sample(
    model: OnnxModel,
    tokens: List[List[int]],
    prompt_tokens: List[List[int]],
    prompt_features: Tensor,
    speed: float = 1.3,
    t_shift: float = 0.5,
    guidance_scale: float = 1.0,
    num_step: int = 16,
    
) -> torch.Tensor:
    # --- Preparation ---
    assert len(tokens) == len(prompt_tokens) == 1
    tokens = torch.tensor(tokens, dtype=torch.int64)
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.int64)
    prompt_features_len = torch.tensor(prompt_features.size(1), dtype=torch.int64)
    speed = torch.tensor(speed, dtype=torch.float32)

    # Run text encoder

    text_condition = model.run_text_encoder(
        tokens, prompt_tokens, prompt_features_len, speed
    )
    batch_size, num_frames, _ = text_condition.shape
    feat_dim = model.feat_dim

    # Get the time schedule
    timesteps = get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=num_step,
        t_shift=t_shift,
    )
    
    # Initialize x with noise (x_0)
    x = torch.randn(batch_size, num_frames, feat_dim)
    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, num_frames - prompt_features.shape[1])
    )
    guidance_scale = torch.tensor(guidance_scale, dtype=torch.float32)

    # --- Sampling Loop ---
    for step in range(num_step):
        t_cur = timesteps[step]
        t_next = timesteps[step + 1]

        # Predict velocity v
        v = model.run_fm_decoder(
            t=t_cur,
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale,
        )

        # Flow matching formula: x_t = (1 - t) * x_0 + t * x_1
        # Therefore: v = x_1 - x_0
        # This implies:
        x_1_pred = x + (1.0 - t_cur) * v
        x_0_pred = x - t_cur * v

        if step < num_step - 1:
            # Anchor-based ODE update for the next step
            x = (1.0 - t_next) * x_0_pred + t_next * x_1_pred
        else:
            # Final step: Snap directly to the predicted clean data (x_1)
            x = x_1_pred

    # Remove the prompt portion from the generated sequence
    x = x[:, prompt_features_len.item() :, :]
    return x

@torch.inference_mode()
def generate_cpu(
    prompt_tokens,
    prompt_features_lens,
    prompt_features,
    prompt_rms,
    text,
    model,
    vocoder,
    tokenizer,
    num_step=4,
    guidance_scale=3.0,
    speed=1.0,
    t_shift=0.9,
    target_rms=0.1,
):
    tokens = tokenizer.texts_to_token_ids([text])
    speed *= 1.3

    pred_features = sample(
        model=model,
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        speed=speed,
        t_shift=t_shift,
        guidance_scale=guidance_scale,
        num_step=num_step,
    )
    # [B, T, C] -> [B, C, T], scale by 10, make contiguous for vocoder
    pred_features = pred_features.transpose(1, 2).mul_(10.0).contiguous()
    wav = vocoder.decode(pred_features).squeeze(1).clamp_(-1.0, 1.0)
    # Volume matching
    wav.mul_(min(prompt_rms / target_rms, 1.0))

    return wav
