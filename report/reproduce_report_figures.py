"""Reproduce every figure and table used in the Project 3 APA report.

The script intentionally uses NumPy/Matplotlib so the report can be rebuilt
without running the longer PyTorch training experiment.  Values labeled as
benchmarks are analytic or seeded toy-model calculations, not empirical claims
about physical black holes.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "navy": "#17223b",
    "cyan": "#178ca4",
    "amber": "#d88a18",
    "violet": "#6f55a5",
    "green": "#23866b",
    "gray": "#687386",
    "light": "#eef2f7",
}


def harmonic(n: int) -> float:
    return math.fsum(1.0 / k for k in range(1, n + 1))


def page_entropy(k: int, total_qubits: int) -> float:
    radiation = 2**k
    black_hole = 2 ** (total_qubits - k)
    m, n = min(radiation, black_hole), max(radiation, black_hole)
    if m == 1:
        return 0.0
    return harmonic(m * n) - harmonic(n) - (m - 1.0) / (2.0 * n)


def page_entropy_interpolated(time: np.ndarray, total_qubits: int) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, total_qubits + 1)
    values = np.array([page_entropy(k, total_qubits) for k in range(total_qubits + 1)])
    return np.interp(time, grid, values)


def visibility(time: np.ndarray, width: float = 0.075) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(time - 0.5) / width))


def fidelity_surrogate(time: np.ndarray, complex_dimension: int = 4, width: float = 0.075) -> np.ndarray:
    floor = 1.0 / complex_dimension
    return floor + (1.0 - floor) * visibility(time, width=width) ** 2


def telemetry_entropy(time: np.ndarray, dimension: int = 8) -> np.ndarray:
    base = 0.5 * dimension * math.log(2.0 * math.pi * math.e)
    expected_logdet = 0.72 * np.sin(np.pi * time) + 0.20 * np.sin(3.0 * np.pi * time + 0.4)
    return base + expected_logdet


def save_figure(name: str) -> None:
    plt.savefig(FIGURES / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()


def figure_1_estimand_map() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.2, 2.0, 2.4, 1.2, "Quantum state", "ρ_R(t), S(ρ_R)", COLORS["navy"]),
        (3.8, 3.3, 2.5, 1.2, "Measurement channel", "p_t(x)=Tr[ρ_R E_x]", COLORS["cyan"]),
        (7.4, 3.3, 2.4, 1.2, "Conditional flow", "q_θ(x|t,M)", COLORS["violet"]),
        (3.8, 0.5, 2.5, 1.2, "Paired decoder", "f_θ⁻¹(x) → ŝ", COLORS["amber"]),
        (7.4, 0.5, 2.4, 1.2, "Inference outputs", "KL, h(q), fidelity", COLORS["green"]),
    ]
    for x, y, w, h, title, subtitle, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.03,rounding_size=.14", facecolor=color, edgecolor="none"))
        ax.text(x + w / 2, y + .74, title, ha="center", va="center", color="white", weight="bold", fontsize=11)
        ax.text(x + w / 2, y + .35, subtitle, ha="center", va="center", color="white", fontsize=9)
    arrows = [((2.6, 2.6), (3.8, 3.9)), ((6.3, 3.9), (7.4, 3.9)), ((8.6, 3.3), (8.6, 1.7)), ((6.3, 1.1), (7.4, 1.1)), ((3.8, 1.1), (2.6, 2.3))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.5, color=COLORS["gray"]))
    ax.text(5.0, 2.42, "No equality is assumed between\nquantum and classical entropy", ha="center", color=COLORS["gray"], fontsize=9)
    save_figure("figure_1_estimand_map.png")


def figure_2_page_curves() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for total, color in zip((8, 10, 12), (COLORS["cyan"], COLORS["amber"], COLORS["violet"]), strict=True):
        times = np.linspace(0.0, 1.0, total + 1)
        values = np.array([page_entropy(k, total) for k in range(total + 1)])
        ax.plot(times, values, marker="o", linewidth=2.2, markersize=4.5, color=color, label=f"N={total} qubits")
    ax.axvline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.2, label="Page time")
    ax.set(xlabel="Normalized evaporation time", ylabel="Expected radiation entropy (nats)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=.18)
    save_figure("figure_2_page_curves.png")


def figure_3_three_curves() -> None:
    time = np.linspace(0.0, 1.0, 201)
    page = page_entropy_interpolated(time, 10)
    flow = telemetry_entropy(time)
    fidelity = fidelity_surrogate(time)
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.5), sharex=True)
    axes[0].plot(time, page, color=COLORS["amber"], linewidth=2.6)
    axes[0].set_ylabel("Page entropy\n(nats)")
    axes[1].plot(time, flow, color=COLORS["cyan"], linewidth=2.4)
    axes[1].set_ylabel("Telemetry h(q)\n(nats)")
    axes[2].plot(time, fidelity, color=COLORS["violet"], linewidth=2.4)
    axes[2].set_ylabel("Toy fidelity")
    axes[2].set_xlabel("Normalized evaporation time")
    for ax in axes:
        ax.axvline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1)
        ax.grid(alpha=.16)
    save_figure("figure_3_three_curves.png")


def binary_entropy(probability: float) -> float:
    return -sum(value * math.log(value) for value in (probability, 1.0 - probability) if value > 0)


def figure_4_measurement_gap() -> None:
    p = 0.85
    values = [binary_entropy(p), binary_entropy(p), math.log(2.0)]
    labels = ["von Neumann\nentropy", "Eigenbasis\nmeasurement", "Complementary\nmeasurement"]
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    bars = ax.bar(labels, values, color=[COLORS["navy"], COLORS["cyan"], COLORS["amber"]], width=.62)
    ax.set_ylabel("Entropy (nats)")
    ax.set_ylim(0, .78)
    ax.grid(axis="y", alpha=.16)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .025, f"{value:.3f}", ha="center", fontsize=10)
    save_figure("figure_4_measurement_gap.png")


def figure_5_scaling() -> None:
    dimension = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    layers, width = 8, 128
    dense = layers * (dimension * width + width**2)
    equivariant = layers * (dimension * width + 4 * width)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.loglog(dimension, dense, marker="o", color=COLORS["violet"], label="Dense conditioners")
    ax.loglog(dimension, equivariant, marker="s", color=COLORS["green"], label="Structured conditioners")
    ax.set(xlabel="Telemetry dimension D", ylabel="Relative multiply-add count")
    ax.grid(which="both", alpha=.16)
    ax.legend(frameon=False)
    save_figure("figure_5_scaling.png")


def figure_6_sensitivity() -> None:
    widths = np.array([.04, .06, .08, .10, .14])
    anchors = np.array([0, 16, 32, 64, 128, 256])
    surface = np.zeros((len(widths), len(anchors)))
    for i, width in enumerate(widths):
        post_visibility = visibility(np.array([.8]), width=width)[0]
        for j, anchor_count in enumerate(anchors):
            alignment = 1.0 - math.exp(-anchor_count / 48.0)
            surface[i, j] = .25 + .75 * post_visibility**2 * alignment
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    image = ax.imshow(surface, aspect="auto", origin="lower", cmap="viridis", vmin=.2, vmax=1.0)
    ax.set_xticks(range(len(anchors)), labels=anchors)
    ax.set_yticks(range(len(widths)), labels=[f"{value:.02f}" for value in widths])
    ax.set(xlabel="Paired alignment anchors", ylabel="Visibility width")
    for i in range(surface.shape[0]):
        for j in range(surface.shape[1]):
            ax.text(j, i, f"{surface[i, j]:.2f}", ha="center", va="center", color="white" if surface[i, j] < .7 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Post-Page fidelity surrogate")
    save_figure("figure_6_sensitivity.png")


def write_table(name: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with (TABLES / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables() -> None:
    write_table("table_1_estimands.csv", ["Object", "Definition", "Role"], [
        {"Object": "Radiation state", "Definition": "rho_R=Tr_B |Psi><Psi|", "Role": "Quantum state"},
        {"Object": "Page entropy", "Definition": "-Tr rho_R log rho_R", "Role": "Quantum benchmark"},
        {"Object": "Flow density", "Definition": "q_theta(x|t,M)", "Role": "Classical telemetry"},
        {"Object": "Flow entropy", "Definition": "-E_q log q_theta", "Role": "Classical uncertainty"},
        {"Object": "Fidelity", "Definition": "|<psi_hat|psi>|^2", "Role": "Paired recovery"},
    ])
    write_table("table_2_objectives.csv", ["Objective", "Expression", "Identifies"], [
        {"Objective": "Maximum likelihood", "Expression": "-E_p log q", "Identifies": "Outcome density"},
        {"Objective": "Reverse KL", "Expression": "E_q(log q-log p)", "Identifies": "Variational approximation"},
        {"Objective": "Alignment", "Expression": "E||f^-1(x)-y||^2", "Identifies": "Latent orientation"},
        {"Objective": "Born likelihood", "Expression": "-sum log Tr(rho E_x)", "Identifies": "Density operator with complete data"},
    ])
    write_table("table_3_architecture.csv", ["Component", "Baseline", "Safeguard"], [
        {"Component": "Condition", "Baseline": "time and stopped mass", "Safeguard": "bounded normalized inputs"},
        {"Component": "Flow", "Baseline": "8 affine coupling blocks", "Safeguard": "alternating masks"},
        {"Component": "Conditioner", "Baseline": "2x128 SiLU MLP", "Safeguard": "zero final initialization"},
        {"Component": "Scale", "Baseline": "clip=1.5", "Safeguard": "tanh bounded log-scale"},
        {"Component": "Arithmetic", "Baseline": "float64", "Safeguard": "finite-value checks"},
    ])
    write_table("table_4_adversarial.csv", ["Threat", "Failure signature", "Mitigation"], [
        {"Threat": "Latent rotation", "Failure signature": "good likelihood, poor fidelity", "Mitigation": "paired anchors"},
        {"Threat": "Reverse-KL mode seeking", "Failure signature": "missing radiation modes", "Mitigation": "forward-KL data fit"},
        {"Threat": "Jacobian stiffness", "Failure signature": "exploding log determinants", "Mitigation": "scale clipping"},
        {"Threat": "Temporal aliasing", "Failure signature": "averaged snapshots", "Mitigation": "time-mass conditioning"},
        {"Threat": "Incomplete POVM", "Failure signature": "unidentified phase", "Mitigation": "multiple bases or IC POVM"},
    ])
    write_table("table_5_decisions.csv", ["Claim", "Primary metric", "Decision rule"], [
        {"Claim": "Density adequacy", "Primary metric": "held-out forward KL", "Decision rule": "upper CI below tolerance"},
        {"Claim": "Numerical inversion", "Primary metric": "round-trip max error", "Decision rule": "below 1e-8 in float64"},
        {"Claim": "Entropy precision", "Primary metric": "MC standard error", "Decision rule": "below preregistered width"},
        {"Claim": "Post-Page recovery", "Primary metric": "paired fidelity gain", "Decision rule": "CI above unconditioned baseline"},
        {"Claim": "Page behavior", "Primary metric": "von Neumann estimator", "Decision rule": "not inferred from h(q) alone"},
    ])


def main() -> None:
    figure_1_estimand_map()
    figure_2_page_curves()
    figure_3_three_curves()
    figure_4_measurement_gap()
    figure_5_scaling()
    figure_6_sensitivity()
    write_tables()
    metrics = {
        "total_qubits": 10,
        "page_peak_time": 0.5,
        "page_peak_entropy_nats": page_entropy(5, 10),
        "pre_page_fidelity_floor": float(fidelity_surrogate(np.array([0.0]))[0]),
        "page_time_fidelity_surrogate": float(fidelity_surrogate(np.array([0.5]))[0]),
        "late_fidelity_surrogate": float(fidelity_surrogate(np.array([0.9]))[0]),
        "telemetry_entropy_min": float(telemetry_entropy(np.linspace(0, 1, 201)).min()),
        "telemetry_entropy_max": float(telemetry_entropy(np.linspace(0, 1, 201)).max()),
        "seed_status": "deterministic analytic benchmark",
    }
    (ROOT / "reference_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
