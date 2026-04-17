import time

import torch

from model_structure.transformer_shared import FQFQNetwork, IQNQNetwork


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 64
GRID_H = 6
GRID_W = 6
BATCH_SIZE = 128
WARMUP = 20
STEPS = 200
NUM_FQF_FRACTIONS = 8
NUM_IQN_QUANTILES = 16


def sync_if_needed():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def benchmark_model(name, model, features, steps, is_fqf):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(WARMUP):
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        loss = output["q_values"].mean()
        if is_fqf:
            loss = loss - 1e-3 * output["fraction_probs"].mean()
        loss.backward()
        optimizer.step()

    sync_if_needed()
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        loss = output["q_values"].mean()
        if is_fqf:
            loss = loss - 1e-3 * output["fraction_probs"].mean()
        loss.backward()
        optimizer.step()
    sync_if_needed()
    elapsed = time.perf_counter() - start

    print(
        f"{name}: total={elapsed:.4f}s | per_step={elapsed / steps * 1000:.3f}ms"
        f" | device={DEVICE}"
    )
    return elapsed


def main():
    torch.manual_seed(0)
    features = torch.randn(BATCH_SIZE, GRID_H * GRID_W, D_MODEL, device=DEVICE)

    fqf = FQFQNetwork(
        d_model=D_MODEL,
        grid_h=GRID_H,
        grid_w=GRID_W,
        num_fractions=NUM_FQF_FRACTIONS,
    ).to(DEVICE)
    iqn = IQNQNetwork(
        d_model=D_MODEL,
        grid_h=GRID_H,
        grid_w=GRID_W,
        num_quantiles=NUM_IQN_QUANTILES,
    ).to(DEVICE)

    fqf_elapsed = benchmark_model("FQF", fqf, features, STEPS, is_fqf=True)
    iqn_elapsed = benchmark_model("IQN", iqn, features, STEPS, is_fqf=False)
    print(f"speedup(FQF/IQN)={fqf_elapsed / iqn_elapsed:.3f}x")


if __name__ == "__main__":
    main()
