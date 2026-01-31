
import torch
import time
import sys
import os

# Add repo root to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from zipvoice.models.modules.solver import DistillEulerSolver, EulerSolver

class DummyModel(torch.nn.Module):
    def __init__(self, feat_dim=100):
        super().__init__()
        self.feat_dim = feat_dim

    def forward_fm_decoder(self, t, xt, text_condition, speech_condition, padding_mask=None, guidance_scale=None, **kwargs):
        if 'x_buffer' in kwargs:
             print("ERROR: x_buffer should not be passed to model function!")
        if 'use_cfg' in kwargs:
             print("ERROR: use_cfg should not be passed to model function!")

        # Mimic the output of the model: velocity v
        # Output shape should match xt: (Batch, SeqLen, FeatDim)

        # Simple deterministic operation for verification:
        # v = xt * 0.1 + text_condition * 0.01 + speech_condition * 0.01 + t * 0.001

        # Handle t shape
        if isinstance(t, float):
             t_val = t
        elif t.dim() == 0:
             t_val = t.item()
        else:
             t_val = t.mean().item() # simplified

        v = xt * 0.1 + text_condition * 0.01 + speech_condition * 0.01 + t_val * 0.001
        return v

def benchmark(name, solver, x, text, speech, mask, num_step, guidance_scale):
    # Warmup
    print(f"--- Benchmarking {name} ---")
    start_time = time.time()
    with torch.no_grad():
        out = solver.sample(
            x=x.clone(),
            text_condition=text,
            speech_condition=speech,
            padding_mask=mask,
            num_step=num_step,
            guidance_scale=guidance_scale
        )
    end_time = time.time()
    print(f"Time: {end_time - start_time:.6f}s")
    return out

def run_test():
    torch.manual_seed(42)
    B = 1
    T = 200 # Medium sequence length
    F = 100 # Feature dim

    x = torch.randn(B, T, F)
    text_condition = torch.randn(B, T, F)
    speech_condition = torch.randn(B, T, F)
    padding_mask = torch.zeros(B, T, dtype=torch.bool) # No padding for simplicity

    model = DummyModel(feat_dim=F)

    # 1. Test DistillEulerSolver (Baseline target)
    print("\n=== Testing DistillEulerSolver ===")
    solver_distill = DistillEulerSolver(model)
    out_distill = benchmark(
        "DistillEulerSolver",
        solver_distill,
        x,
        text_condition,
        speech_condition,
        padding_mask,
        num_step=10,
        guidance_scale=3.0
    )

    # Check against baseline
    if os.path.exists("tests/baseline_output.pt"):
        out_baseline = torch.load("tests/baseline_output.pt")
        diff = torch.abs(out_distill - out_baseline)
        max_diff = diff.max().item()
        print(f"Distill Max difference vs Baseline: {max_diff:.9f}")
        if max_diff < 1e-5:
            print("SUCCESS: Distill Output matches baseline.")
        else:
            print("WARNING: Distill Output differs!")

    # 2. Test EulerSolver (Base class, exercised DiffusionModel changes)
    print("\n=== Testing EulerSolver (Base) ===")
    # EulerSolver uses DiffusionModel which does splitting/catting.
    # We don't have a baseline for this (unless we generate one from 'before' code,
    # but we already modified the code).
    # We just want to make sure it runs without error and produces output.
    solver_base = EulerSolver(model)
    out_base = benchmark(
        "EulerSolver",
        solver_base,
        x,
        text_condition,
        speech_condition,
        padding_mask,
        num_step=10,
        guidance_scale=3.0
    )
    print("EulerSolver ran successfully.")

    # Basic sanity check: Shape should be same
    if out_base.shape == x.shape:
        print("SUCCESS: EulerSolver output shape correct.")
    else:
        print(f"ERROR: EulerSolver output shape mismatch: {out_base.shape}")

if __name__ == "__main__":
    run_test()
