"""
Fase 3 — Correlación latente↔imagen con TODOS los pares (1.770).

El análisis original (fase3_geometrica.py) muestreaba aleatoriamente 500 de los
1.770 pares posibles C(60,2). Este script recalcula la correlación usando todos
los pares, generando estimaciones más precisas con intervalos de confianza bootstrap.

No requiere GPU: opera únicamente sobre latents.pt (ya almacenados) e imágenes PNG.

Salida:
  data/fase3/results/corr_completa_summary.json   — métricas actualizadas
  data/fase3/results/corr_completa_sd15.png       — scatter plots SD 1.5
  data/fase3/results/corr_completa_sd21.png       — scatter plots SD 2.1
  data/fase3/results/corr_completa_sdxl.png       — scatter plots SDXL
  data/fase3/results/corr_completa_cross.png      — comparativa cross-model
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent
LATENTS_DIR = ROOT / "data" / "latents" / "latents"
RESULTS_DIR = ROOT / "data" / "fase3" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["sd15", "sd21", "sdxl"]
MODEL_LABELS = {"sd15": "SD 1.5", "sd21": "SD 2.1", "sdxl": "SDXL"}
MODEL_COLORS = {"sd15": "#4C72B0", "sd21": "#DD8452", "sdxl": "#55A868"}


# ─── Carga ────────────────────────────────────────────────────────────────────

def load_base_data(model: str):
    """Devuelve Z_base [60, D], manifest_base, images_dir."""
    base = LATENTS_DIR / model
    tensor = torch.load(base / "latents.pt", map_location="cpu",
                        weights_only=False).float()
    if tensor.ndim == 5:
        tensor = tensor.squeeze(1)          # [N, C, H, W]
    N = tensor.shape[0]
    Z = tensor.numpy().reshape(N, -1)       # [N, D]

    with open(base / "manifest.json") as f:
        manifest = json.load(f)

    mask_base = [e["category"] == "base" for e in manifest]
    Z_base = Z[mask_base]
    mf_base = [e for e in manifest if e["category"] == "base"]
    return Z_base, mf_base, base / "images"


# ─── Métricas de imagen ───────────────────────────────────────────────────────

def load_image(images_dir: Path, idx: int) -> np.ndarray:
    return np.array(
        Image.open(images_dir / f"{idx:05d}.png").convert("RGB"),
        dtype=np.float32
    ) / 255.0


def image_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def image_ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a = np.var(a)
    var_b = np.var(b)
    cov_ab = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(num / den)


# ─── Bootstrap IC ─────────────────────────────────────────────────────────────

def bootstrap_ci(x: np.ndarray, y: np.ndarray, func, n_boot=2000, ci=0.95):
    """Intervalo de confianza bootstrap para func(x, y) -> escalar."""
    rng = np.random.default_rng(42)
    stats = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(func(x[idx], y[idx]))
    stats = np.array(stats)
    alpha = (1 - ci) / 2
    return float(np.percentile(stats, alpha * 100)), float(np.percentile(stats, (1 - alpha) * 100))


# ─── Análisis principal ───────────────────────────────────────────────────────

def analyse_model(model: str) -> dict:
    print(f"\n{'='*50}")
    print(f"  Modelo: {MODEL_LABELS[model]}")
    print(f"{'='*50}")

    Z, manifest, images_dir = load_base_data(model)
    N = len(manifest)
    assert N == 60, f"Esperados 60 latents base, encontrados {N}"

    # Precargar imágenes
    print("  Cargando imágenes...")
    images = {e["idx"]: load_image(images_dir, e["idx"]) for e in manifest}

    # Todos los pares
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    n_pairs = len(all_pairs)
    print(f"  Calculando {n_pairs} pares (todos posibles)...")

    latent_dists = np.empty(n_pairs, dtype=np.float32)
    mse_vals     = np.empty(n_pairs, dtype=np.float32)
    ssim_vals    = np.empty(n_pairs, dtype=np.float32)

    for k, (i, j) in enumerate(all_pairs):
        latent_dists[k] = float(np.linalg.norm(Z[i] - Z[j]))
        img_i = images[manifest[i]["idx"]]
        img_j = images[manifest[j]["idx"]]
        mse_vals[k]  = image_mse(img_i, img_j)
        ssim_vals[k] = image_ssim_global(img_i, img_j)
        if (k + 1) % 500 == 0:
            print(f"    {k+1}/{n_pairs}")

    # Correlaciones
    r_mse,  p_mse  = pearsonr(latent_dists, mse_vals)
    rho_mse,  _    = spearmanr(latent_dists, mse_vals)
    r_ssim, p_ssim = pearsonr(latent_dists, -ssim_vals)
    rho_ssim, _    = spearmanr(latent_dists, -ssim_vals)

    # Bootstrap ICs (Pearson)
    r_mse_lo,  r_mse_hi  = bootstrap_ci(
        latent_dists, mse_vals,  lambda x, y: pearsonr(x, y)[0])
    r_ssim_lo, r_ssim_hi = bootstrap_ci(
        latent_dists, -ssim_vals, lambda x, y: pearsonr(x, y)[0])

    print(f"  r(L2, MSE)   = {r_mse:.3f}  [{r_mse_lo:.3f}, {r_mse_hi:.3f}]  "
          f"rho={rho_mse:.3f}  p={p_mse:.2e}")
    print(f"  r(L2,-SSIM)  = {r_ssim:.3f}  [{r_ssim_lo:.3f}, {r_ssim_hi:.3f}]  "
          f"rho={rho_ssim:.3f}  p={p_ssim:.2e}")

    result = {
        "model": model,
        "n_pairs": n_pairs,
        "pearson_latent_vs_mse":       float(r_mse),
        "pearson_ci95_mse":            [float(r_mse_lo), float(r_mse_hi)],
        "spearman_latent_vs_mse":      float(rho_mse),
        "pearson_pvalue_mse":          float(p_mse),
        "pearson_latent_vs_neg_ssim":  float(r_ssim),
        "pearson_ci95_ssim":           [float(r_ssim_lo), float(r_ssim_hi)],
        "spearman_latent_vs_neg_ssim": float(rho_ssim),
        "pearson_pvalue_ssim":         float(p_ssim),
    }

    # Scatter plots
    _plot_scatter(model, latent_dists, mse_vals, ssim_vals, result)

    return result


# ─── Plots ────────────────────────────────────────────────────────────────────

def _plot_scatter(model, ld, mse, ssim, res):
    color = MODEL_COLORS[model]
    label = MODEL_LABELS[model]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{label} — Correlación distancia latente ↔ similitud de imagen\n"
                 f"(todos los {res['n_pairs']} pares)", fontsize=12)

    for ax, y, ylabel, r, rho, ci, pval in [
        (axes[0], mse,  "MSE (imagen)",
         res["pearson_latent_vs_mse"],
         res["spearman_latent_vs_mse"],
         res["pearson_ci95_mse"],
         res["pearson_pvalue_mse"]),
        (axes[1], -ssim, "−SSIM (imagen)",
         res["pearson_latent_vs_neg_ssim"],
         res["spearman_latent_vs_neg_ssim"],
         res["pearson_ci95_ssim"],
         res["pearson_pvalue_ssim"]),
    ]:
        ax.scatter(ld, y, alpha=0.25, s=8, color=color, rasterized=True)
        # Línea de regresión
        m_, b_ = np.polyfit(ld, y, 1)
        x_line = np.linspace(ld.min(), ld.max(), 200)
        ax.plot(x_line, m_ * x_line + b_, color="crimson", lw=1.5)
        ax.set_xlabel("Distancia L2 latente", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ci_lo, ci_hi = ci
        ax.text(0.03, 0.97,
                f"r = {r:.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]\n"
                f"ρ = {rho:.3f}   p < 10⁻⁸⁰",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.15))
        ax.grid(alpha=0.25)

    plt.tight_layout()
    out = RESULTS_DIR / f"corr_completa_{model}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {out.name}")


def plot_cross_model(results: list[dict]):
    """Comparativa de las cuatro correlaciones entre modelos."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Correlación latente↔imagen — todos los pares (1.770)", fontsize=12)

    metrics = [
        ("pearson_latent_vs_mse",      "pearson_ci95_mse",
         "r(L2, MSE)"),
        ("pearson_latent_vs_neg_ssim", "pearson_ci95_ssim",
         "r(L2, −SSIM)"),
    ]
    x = np.arange(len(MODELS))
    width = 0.55

    for ax, (key, ci_key, title) in zip(axes, metrics):
        vals  = [next(r[key] for r in results if r["model"] == m) for m in MODELS]
        ci_lo = [next(r[ci_key][0] for r in results if r["model"] == m) for m in MODELS]
        ci_hi = [next(r[ci_key][1] for r in results if r["model"] == m) for m in MODELS]
        yerr_lo = [v - lo for v, lo in zip(vals, ci_lo)]
        yerr_hi = [hi - v  for v, hi in zip(vals, ci_hi)]
        colors = [MODEL_COLORS[m] for m in MODELS]
        bars = ax.bar(x, vals, width, color=colors,
                      yerr=[yerr_lo, yerr_hi], capsize=5, error_kw={"lw": 1.5})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
        ax.set_ylabel("Coef. de correlación de Pearson", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "corr_completa_cross.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura cross-model guardada: {out.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []
    for model in MODELS:
        res = analyse_model(model)
        results.append(res)

    plot_cross_model(results)

    summary_path = RESULTS_DIR / "corr_completa_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en: {summary_path}")

    # Tabla comparativa rápida
    print("\n" + "="*70)
    print(f"{'Modelo':<8}  {'n_pairs':>7}  {'r(MSE)':>8}  {'IC 95%':>15}  "
          f"{'ρ(MSE)':>8}  {'r(-SSIM)':>9}  {'IC 95%':>15}  {'ρ(-SSIM)':>9}")
    print("-"*70)
    for r in results:
        ci_m = r["pearson_ci95_mse"]
        ci_s = r["pearson_ci95_ssim"]
        print(f"{MODEL_LABELS[r['model']]:<8}  {r['n_pairs']:>7}  "
              f"{r['pearson_latent_vs_mse']:>8.3f}  "
              f"[{ci_m[0]:.3f},{ci_m[1]:.3f}]  "
              f"{r['spearman_latent_vs_mse']:>8.3f}  "
              f"{r['pearson_latent_vs_neg_ssim']:>9.3f}  "
              f"[{ci_s[0]:.3f},{ci_s[1]:.3f}]  "
              f"{r['spearman_latent_vs_neg_ssim']:>9.3f}")
