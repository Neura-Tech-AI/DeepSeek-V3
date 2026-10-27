#!/usr/bin/env python3

import argparse
from itertools import product


def calc(c):
    V = c["vocab_size"]
    D = c["dim"]
    I = c["inter_dim"]
    MI = c["moe_inter_dim"]
    L = c["n_layers"]
    DL = c["n_dense_layers"]
    H = c["n_heads"]

    E = c["n_routed_experts"]
    SE = c["n_shared_experts"]
    K = c["n_activated_experts"]

    QR = c["q_lora_rank"]
    KR = c["kv_lora_rank"]

    NOPE = c["qk_nope_head_dim"]
    ROPE = c["qk_rope_head_dim"]
    VD = c["v_head_dim"]

    # ---------------------------------------------------------
    # Embedding + LM Head
    # ---------------------------------------------------------

    embedding = V * D
    lm_head = D * V

    # ---------------------------------------------------------
    # MLA
    # Matches inference/model.py
    # ---------------------------------------------------------

    q_head_dim = NOPE + ROPE

    if QR == 0:
        wq = D * (H * q_head_dim)
    else:
        wq_a = D * QR
        wq_b = QR * (H * q_head_dim)
        q_norm = QR

        wq = wq_a + wq_b + q_norm

    wkv_a = D * (KR + ROPE)
    kv_norm = KR

    wkv_b = KR * (H * (NOPE + VD))

    wo = (H * VD) * D

    mla = wq + wkv_a + kv_norm + wkv_b + wo

    # ---------------------------------------------------------
    # RMSNorm
    # Two per transformer block
    # ---------------------------------------------------------

    block_norms = 2 * D

    # ---------------------------------------------------------
    # Dense MLP
    # w1 + w2 + w3
    # ---------------------------------------------------------

    dense_mlp = 3 * D * I

    # ---------------------------------------------------------
    # One MoE Expert
    # w1 + w2 + w3
    # ---------------------------------------------------------

    one_expert = 3 * D * MI

    # ---------------------------------------------------------
    # MoE Gate
    # ---------------------------------------------------------

    gate = E * D

    # DeepSeek-V3 code adds gate bias only when D == 7168
    gate_bias = E if D == 7168 else 0

    gate += gate_bias

    # ---------------------------------------------------------
    # Shared Expert
    # shared_experts = MLP(D, SE * MI)
    # ---------------------------------------------------------

    shared_expert = 3 * D * (SE * MI)

    # ---------------------------------------------------------
    # Complete MoE layer
    # ---------------------------------------------------------

    all_routed_experts = E * one_expert

    moe_total = (
        gate
        + all_routed_experts
        + shared_expert
    )

    moe_layers = L - DL

    # ---------------------------------------------------------
    # TOTAL PARAMETERS
    # ---------------------------------------------------------

    total = (
        embedding
        + lm_head
        + D                         # final RMSNorm
        + DL * (
            mla
            + block_norms
            + dense_mlp
        )
        + moe_layers * (
            mla
            + block_norms
            + moe_total
        )
    )

    # ---------------------------------------------------------
    # ACTIVE PARAMETERS / TOKEN
    #
    # Dense layers:
    # MLA + norms + dense MLP
    #
    # MoE layers:
    # MLA + norms + gate
    # + K routed experts
    # + shared expert
    # ---------------------------------------------------------

    active_dense = (
        mla
        + block_norms
        + dense_mlp
    )

    active_moe = (
        mla
        + block_norms
        + gate
        + K * one_expert
        + shared_expert
    )

    active = (
        DL * active_dense
        + moe_layers * active_moe
        + D
    )

    return total, active


def make_config(
    D,
    L,
    E,
    K,
    MI,
    H,
    dense_layers,
):
    return {
        "vocab_size": 32000,

        "dim": D,

        # Approximate dense FFN ratio.
        # Can be manually changed later.
        "inter_dim": int(D * 2.75),

        "moe_inter_dim": MI,

        "n_layers": L,
        "n_dense_layers": dense_layers,

        "n_heads": H,

        "n_routed_experts": E,
        "n_shared_experts": 1,
        "n_activated_experts": K,

        # DeepSeek-style MLA ranks
        "q_lora_rank": max(128, D // 4),
        "kv_lora_rank": max(64, D // 8),

        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 32,
        "v_head_dim": 64,
    }


def search(target_m):
    target = target_m * 1_000_000

    dims = [
        768,
        896,
        1024,
        1152,
        1280,
        1408,
        1536,
    ]

    layers = [
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        24,
        28,
        32,
    ]

    experts = [
        4,
        8,
        16,
        32,
    ]

    topks = [
        1,
        2,
        4,
    ]

    expert_dims = [
        256,
        384,
        512,
        640,
        768,
        1024,
    ]

    dense_layers = [
        1,
        2,
        3,
    ]

    results = []

    for D, L, E, K, MI, DL in product(
        dims,
        layers,
        experts,
        topks,
        expert_dims,
        dense_layers,
    ):

        if K > E:
            continue

        for H in [8, 12, 16, 24, 32]:

            if D % H != 0:
                continue

            config = make_config(
                D=D,
                L=L,
                E=E,
                K=K,
                MI=MI,
                H=H,
                dense_layers=DL,
            )

            total, active = calc(config)

            error = abs(total - target)

            results.append(
                (
                    error,
                    total,
                    active,
                    config,
                )
            )

    results.sort(key=lambda x: x[0])

    print()
    print("=" * 100)
    print(f"       EXACT DEEPSEEK-STYLE SEARCH: ~{target_m:.0f}M PARAMETERS")
    print("=" * 100)

    print(
        f"{'#':<3}"
        f"{'Total':>11}"
        f"{'Active':>11}"
        f"{'Dim':>7}"
        f"{'Layers':>8}"
        f"{'Dense':>7}"
        f"{'Experts':>9}"
        f"{'TopK':>7}"
        f"{'ExpDim':>9}"
        f"{'Heads':>7}"
    )

    print("-" * 100)

    for i, (_, total, active, c) in enumerate(results[:20], 1):

        print(
            f"{i:<3}"
            f"{total / 1e6:>10.2f}M"
            f"{active / 1e6:>10.2f}M"
            f"{c['dim']:>7}"
            f"{c['n_layers']:>8}"
            f"{c['n_dense_layers']:>7}"
            f"{c['n_routed_experts']:>9}"
            f"{c['n_activated_experts']:>7}"
            f"{c['moe_inter_dim']:>9}"
            f"{c['n_heads']:>7}"
        )

    print("=" * 100)

    # Best configuration
    best = results[0]

    _, total, active, c = best

    print()
    print("BEST CONFIGURATION")
    print("-" * 50)

    for key, value in c.items():
        print(f"{key:<22}: {value}")

    print()
    print(f"TOTAL PARAMETERS : {total:,}")
    print(f"TOTAL             : {total / 1e6:.3f}M")
    print(f"ACTIVE / TOKEN    : {active:,}")
    print(f"ACTIVE            : {active / 1e6:.3f}M")

    print("-" * 50)


def show_default():

    config = make_config(
        D=1024,
        L=12,
        E=8,
        K=2,
        MI=512,
        H=16,
        dense_layers=2,
    )

    total, active = calc(config)

    print()
    print("DEFAULT CONFIG")
    print("-" * 40)
    print(f"Total  : {total:,} ({total / 1e6:.2f}M)")
    print(f"Active : {active:,} ({active / 1e6:.2f}M)")
    print("-" * 40)


parser = argparse.ArgumentParser(
    description="DeepSeek-V3 style Neuron parameter calculator"
)

parser.add_argument(
    "--target",
    type=float,
    help="Target parameter count in millions"
)

args = parser.parse_args()

if args.target:
    search(args.target)
else:
    show_default()
