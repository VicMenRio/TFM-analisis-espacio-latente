"""
Fase 3 — Análisis geométrico del espacio latente.

Hipótesis central: imágenes perceptualmente similares deben tener latents más
próximos que imágenes perceptualmente distintas. Se verifica esta propiedad
para los tres modelos y se comparan sus estructuras de distancias.

Análisis realizados (todos locales, sin GPU):
  1. Distancias L2 intra-prompt vs inter-prompt
     — ¿los latents del mismo prompt con distinta seed están más cerca entre
       sí que los de prompts distintos?

  2. Distribución de distancias por modelo
     — ¿qué tan disperso es el espacio latente? ¿cómo varía entre modelos?

  3. Correlación distancia latente ↔ similitud de imagen
     — usando MSE y similitud de correlación pixel como proxy perceptual.
       (LPIPS requiere GPU; se deja para Colab en Fase 5+)

  4. Ratio de separabilidad intra/inter por modelo
     — métrica escalar que cuantifica cuánto se agrupan los latents del
       mismo prompt.

  5. Comparativa cross-model
     — normalización de distancias, distribuciones, separabilidad.

Entrada:
  corpus/latents/{sd15,sd21,sdxl}/latents.pt   [N, 1, C, H, W]
  corpus/latents/{sd15,sd21,sdxl}/manifest.json
  corpus/latents/{sd15,sd21,sdxl}/images/       *.png  (512×512 RGB)

Salida:
  analisis_geometrico/resultados/geometrica_summary.json
  analisis_geometrico/resultados/geo_{model}_distancias.png
  analisis_geometrico/resultados/geo_{model}_correlacion.png
  analisis_geometrico/resultados/geo_cross_model.png
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr, ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent
LATENTS_DIR = ROOT / "corpus" / "latents"
RESULTS_DIR = ROOT / "analisis_geometrico" / "resultados"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["sd15", "sd21", "sdxl"]
MODEL_COLORS = {"sd15": "#4C72B0", "sd21": "#DD8452", "sdxl": "#55A868"}


# ─── Carga ───────────────────────────────────────────────────────────────────

def load_data(model: str):
    """Devuelve (Z_base, Z_attr, manifest_base, manifest_attr, images_dir)."""
    base = LATENTS_DIR / model
    tensor = torch.load(base / "latents.pt", map_location="cpu", weights_only=False).float()
    if tensor.ndim == 5:
        tensor = tensor.squeeze(1)           # [N, C, H, W]
    N = tensor.shape[0]
    Z = tensor.numpy().reshape(N, -1)        # [N, D]

    with open(base / "manifest.json") as f:
        manifest = json.load(f)

    mask_base = np.array([e["category"] == "base" for e in manifest])
    Z_base = Z[mask_base]
    Z_attr = Z[~mask_base]
    mf_base = [e for e in manifest if e["category"] == "base"]
    mf_attr = [e for e in manifest if e["category"] != "base"]

    return Z_base, Z_attr, mf_base, mf_attr, base / "images"


# ─── Similitud de imagen ──────────────────────────────────────────────────────

def load_image(images_dir: Path, idx: int) -> np.ndarray:
    path = images_dir / f"{idx:05d}.png"
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def image_mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def image_ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM global (sin ventana) como proxy de similitud perceptual."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a = np.var(a)
    var_b = np.var(b)
    cov_ab = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(num / den)


# ─── Análisis de distancias ───────────────────────────────────────────────────

def compute_distance_matrix(Z: np.ndarray) -> np.ndarray:
    """Matriz de distancias L2 euclídeas [N, N]. Usa la identidad ||x-y||²=||x||²+||y||²-2x·y
    para evitar crear el tensor [N, N, D] completo en memoria."""
    # La identidad ||x-y||² = ||x||² + ||y||² - 2xᵀy permite calcular toda la
    # matriz en tres operaciones vectorizadas (O(N²D)) en lugar de expandir el
    # tensor [N,N,D] de diferencias (O(N²D) en memoria). Para D=65.536 (SDXL) y
    # N=60, la expansión directa requeriría ~7 GB; este enfoque usa solo ~28 MB.
    sq_norms = (Z ** 2).sum(axis=1)                     # [N]
    gram = Z @ Z.T                                       # [N, N]
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * gram
    return np.sqrt(np.clip(sq_dists, 0, None))


def intra_inter_distances(D_mat: np.ndarray, manifest: list[dict]):
    """
    Separa distancias en intra-prompt (mismo id, distinta seed) e inter-prompt.
    Devuelve (intra_dists, inter_dists).
    """
    N = len(manifest)
    intra, inter = [], []
    for i in range(N):
        for j in range(i + 1, N):
            d = D_mat[i, j]
            if manifest[i]["id"] == manifest[j]["id"]:
                intra.append(d)
            else:
                inter.append(d)
    return np.array(intra), np.array(inter)


def separability_ratio(intra: np.ndarray, inter: np.ndarray) -> float:
    """
    Ratio de separabilidad: inter_mean / intra_mean.
    Mayor → el espacio agrupa mejor las variaciones del mismo prompt.
    """
    # Ratio > 1: los latentes de distintos prompts están más separados que los del
    # mismo prompt con distintas semillas → el contenido semántico domina sobre la
    # variabilidad estocástica del proceso de denoising.
    # Ratio < 1 (observado en SD 1.5 y SD 2.1): la variabilidad estocástica es
    # comparable o superior a la señal semántica, lo que penaliza la edición en z₀.
    return float(inter.mean() / (intra.mean() + 1e-8))


# ─── Correlación latente ↔ imagen ─────────────────────────────────────────────

def compute_latent_image_correlation(
    Z: np.ndarray,
    manifest: list[dict],
    images_dir: Path,
    max_pairs: int = 500,
) -> dict:
    """
    Muestrea pares de índices, calcula distancia L2 en latent y MSE/SSIM
    en imagen, y mide la correlación (Pearson, Spearman).
    Precarga todas las imágenes en memoria para evitar I/O repetido por par.
    """
    N = len(manifest)
    # Precargar todas las imágenes de una vez (512×512×3 × 60 ≈ 45 MB)
    print("      Precargando imágenes...")
    images = {e["idx"]: load_image(images_dir, e["idx"]) for e in manifest}

    # Generar todos los pares superiores, muestrar si hay demasiados
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng = np.random.default_rng(42)
    if len(all_pairs) > max_pairs:
        sel = rng.choice(len(all_pairs), max_pairs, replace=False)
        pairs = [all_pairs[k] for k in sel]
    else:
        pairs = all_pairs

    latent_dists, mse_vals, ssim_vals = [], [], []

    for i, j in pairs:
        d_lat = float(np.linalg.norm(Z[i] - Z[j]))
        img_i = images[manifest[i]["idx"]]
        img_j = images[manifest[j]["idx"]]
        mse = image_mse(img_i, img_j)
        ssim = image_ssim_global(img_i, img_j)
        latent_dists.append(d_lat)
        mse_vals.append(mse)
        ssim_vals.append(ssim)

    ld = np.array(latent_dists)
    mse = np.array(mse_vals)
    ssim = np.array(ssim_vals)

    r_mse, p_mse = pearsonr(ld, mse)
    rho_mse, _ = spearmanr(ld, mse)
    r_ssim, p_ssim = pearsonr(ld, -ssim)   # -ssim para que mayor latent dist ↔ menor similitud
    rho_ssim, _ = spearmanr(ld, -ssim)

    return {
        "n_pairs": len(pairs),
        "pearson_latent_vs_mse": float(r_mse),
        "spearman_latent_vs_mse": float(rho_mse),
        "pearson_pvalue_mse": float(p_mse),
        "pearson_latent_vs_neg_ssim": float(r_ssim),
        "spearman_latent_vs_neg_ssim": float(rho_ssim),
        "pearson_pvalue_ssim": float(p_ssim),
        # para plots
        "_latent_dists": ld,
        "_mse_vals": mse,
        "_ssim_vals": ssim,
    }


# ─── Dimensionalidad efectiva ─────────────────────────────────────────────────

def participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    Participation ratio: (∑λ)² / ∑λ² — número efectivo de dimensiones activas.
    """
    lam = eigenvalues[eigenvalues > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def effective_rank(Z: np.ndarray) -> dict:
    """Calcula participation ratio y dimensión efectiva a 80/90% de varianza.
    Usa SVD de la matriz de datos centrada (O(N²D)) en lugar de la eigendescomposición
    de la covarianza (O(D³)), lo que es viable incluso para D=65536."""
    # El Participation Ratio (PR = (Σλₖ)² / Σλₖ²) cuantifica el número efectivo
    # de dimensiones activas de forma continua y sin necesidad de elegir un umbral
    # de varianza explicada. Para una distribución perfectamente isótropa, PR = N;
    # para una distribución concentrada en una dimensión, PR = 1.
    # La SVD thin (full_matrices=False) sobre la matriz centrada [N×D] devuelve
    # vectores singulares de dimensión N en lugar de D, lo que es viable incluso
    # para D=65.536 con N=60 (SDXL). Los valores propios de la covarianza son S²/(N-1).
    Z_c = Z - Z.mean(axis=0)
    # SVD truncada: para N<<D, full_matrices=False da U[N,N], S[N], Vt[N,D]
    _, S, _ = np.linalg.svd(Z_c, full_matrices=False)
    eigvals = (S ** 2) / max(len(Z) - 1, 1)   # valores propios de la covarianza
    eigvals = eigvals[eigvals > 0]
    pr = participation_ratio(eigvals)
    var_cum = eigvals.cumsum() / eigvals.sum()
    k80 = int(np.searchsorted(var_cum, 0.80)) + 1
    k90 = int(np.searchsorted(var_cum, 0.90)) + 1
    k95 = int(np.searchsorted(var_cum, 0.95)) + 1
    return {
        "participation_ratio": pr,
        "dims_80pct_variance": k80,
        "dims_90pct_variance": k90,
        "dims_95pct_variance": k95,
        "total_variance": float(eigvals.sum()),
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_distancias(D_mat, manifest, intra, inter, model):
    fig = plt.figure(figsize=(15, 4))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    fig.suptitle(f"{model.upper()} — Distancias en el espacio latente", fontsize=13)
    color = MODEL_COLORS[model]

    # Mapa de calor de la matriz de distancias (base)
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(D_mat, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    # Líneas de separación entre prompts (cada 3 seeds)
    for tick in range(3, D_mat.shape[0], 3):
        ax1.axhline(tick - 0.5, color="white", lw=0.5, alpha=0.5)
        ax1.axvline(tick - 0.5, color="white", lw=0.5, alpha=0.5)
    ax1.set_title("Matriz de distancias L2\n(prompts base, 3 seeds c/u)")
    ax1.set_xlabel("Índice latent"); ax1.set_ylabel("Índice latent")

    # Distribución intra vs inter
    ax2 = fig.add_subplot(gs[1])
    bins = 40
    ax2.hist(intra, bins=bins, alpha=0.7, label="Intra-prompt", color=color, density=True)
    ax2.hist(inter, bins=bins, alpha=0.5, label="Inter-prompt", color="gray", density=True)
    ax2.axvline(intra.mean(), color=color, lw=2, linestyle="--", alpha=0.8)
    ax2.axvline(inter.mean(), color="gray", lw=2, linestyle="--", alpha=0.8)
    ks, pval = ks_2samp(intra, inter)
    sep = inter.mean() / (intra.mean() + 1e-8)
    ax2.set_xlabel("Distancia L2"); ax2.set_ylabel("Densidad")
    ax2.set_title("Intra-prompt vs inter-prompt")
    ax2.text(0.97, 0.97,
             f"KS={ks:.3f}  p={pval:.1e}\nsep ratio={sep:.2f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", alpha=0.15))
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # Boxplot intra vs inter (resumen compacto)
    ax3 = fig.add_subplot(gs[2])
    ax3.boxplot([intra, inter], tick_labels=["Intra\n(mismo prompt)", "Inter\n(prompt distinto)"],
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.5),
                medianprops=dict(color="black", lw=2))
    ax3.set_ylabel("Distancia L2")
    ax3.set_title("Distribución de distancias")
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / f"geo_{model}_distancias.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def plot_correlacion(corr: dict, model: str):
    ld = corr["_latent_dists"]
    mse = corr["_mse_vals"]
    ssim = corr["_ssim_vals"]
    color = MODEL_COLORS[model]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{model.upper()} — Distancia latente vs similitud de imagen", fontsize=13)

    ax = axes[0]
    ax.scatter(ld, mse, alpha=0.35, s=15, color=color)
    z = np.polyfit(ld, mse, 1)
    x_line = np.linspace(ld.min(), ld.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r--", lw=1.5)
    ax.set_xlabel("Distancia L2 latente"); ax.set_ylabel("MSE imagen")
    ax.set_title(f"Dist. latente vs MSE imagen\nr={corr['pearson_latent_vs_mse']:.3f}  "
                 f"ρ={corr['spearman_latent_vs_mse']:.3f}")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(ld, ssim, alpha=0.35, s=15, color=color)
    z2 = np.polyfit(ld, ssim, 1)
    ax.plot(x_line, np.polyval(z2, x_line), "r--", lw=1.5)
    ax.set_xlabel("Distancia L2 latente"); ax.set_ylabel("SSIM imagen (↑ = más similar)")
    ax.set_title(f"Dist. latente vs SSIM imagen\nr={-corr['pearson_latent_vs_neg_ssim']:.3f}  "
                 f"ρ={-corr['spearman_latent_vs_neg_ssim']:.3f}")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / f"geo_{model}_correlacion.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def plot_cross_model(all_results: dict):
    models = list(all_results.keys())
    colors = [MODEL_COLORS[m] for m in models]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Fase 3 — Comparativa geométrica cross-model", fontsize=14)

    # Distribución de distancias inter-prompt por modelo
    ax = axes[0][0]
    for m, c in zip(models, colors):
        ax.hist(all_results[m]["_inter_dists"], bins=40, alpha=0.5, label=m.upper(),
                color=c, density=True)
    ax.set_xlabel("Distancia L2 inter-prompt"); ax.set_ylabel("Densidad")
    ax.set_title("Distribución distancias inter-prompt"); ax.legend(); ax.grid(alpha=0.3)

    # Ratio de separabilidad
    ax = axes[0][1]
    seps = [all_results[m]["separability_ratio"] for m in models]
    bars = ax.bar(models, seps, color=colors, alpha=0.85)
    for bar, v in zip(bars, seps):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(seps) * 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Ratio de separabilidad\n(inter_μ / intra_μ  — mayor es mejor)")
    ax.set_ylabel("Ratio"); ax.grid(axis="y", alpha=0.3)

    # Pearson latent vs MSE
    ax = axes[0][2]
    corrs = [all_results[m]["correlation"]["pearson_latent_vs_mse"] for m in models]
    bars = ax.bar(models, corrs, color=colors, alpha=0.85)
    for bar, v in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(corrs) * 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Correlación Pearson\ndist. latente ↔ MSE imagen")
    ax.set_ylabel("r"); ax.grid(axis="y", alpha=0.3)

    # Dimensionalidad efectiva
    ax = axes[1][0]
    pr_vals = [all_results[m]["effective_rank"]["participation_ratio"] for m in models]
    bars = ax.bar(models, pr_vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, pr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(pr_vals) * 0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Participation ratio\n(dimensionalidad efectiva)")
    ax.set_ylabel("PR"); ax.grid(axis="y", alpha=0.3)

    # Dimensiones para 80/90/95% de varianza
    ax = axes[1][1]
    k80 = [all_results[m]["effective_rank"]["dims_80pct_variance"] for m in models]
    k90 = [all_results[m]["effective_rank"]["dims_90pct_variance"] for m in models]
    k95 = [all_results[m]["effective_rank"]["dims_95pct_variance"] for m in models]
    x = np.arange(len(models)); w = 0.25
    ax.bar(x - w, k80, width=w, label="80%", color="#aec7e8", alpha=0.9)
    ax.bar(x,     k90, width=w, label="90%", color="#6baed6", alpha=0.9)
    ax.bar(x + w, k95, width=w, label="95%", color="#2171b5", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("Nº dimensiones"); ax.set_title("Dims. para capturar X% de varianza")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # Tabla resumen
    ax = axes[1][2]
    ax.axis("off")
    rows = []
    headers = ["Modelo", "Intra μ", "Inter μ", "Sep.", "PR", "r(MSE)"]
    for m in models:
        r = all_results[m]
        rows.append([
            m.upper(),
            f"{r['intra_mean']:.1f}",
            f"{r['inter_mean']:.1f}",
            f"{r['separability_ratio']:.2f}",
            f"{r['effective_rank']['participation_ratio']:.1f}",
            f"{r['correlation']['pearson_latent_vs_mse']:.3f}",
        ])
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.8)
    ax.set_title("Resumen métricas geométricas", pad=14)

    plt.tight_layout()
    out = RESULTS_DIR / "geo_cross_model.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    geo_cross_model.png guardado.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_model(model: str) -> dict | None:
    print(f"\n  [{model.upper()}]")
    try:
        Z_base, _, mf_base, _, images_dir = load_data(model)
    except FileNotFoundError as e:
        print(f"    SKIP: {e}"); return None

    D = Z_base.shape[1]
    N = Z_base.shape[0]
    print(f"    {N} latents base  |  dim={D}")

    # ── Análisis geométrico en cuatro pasos ─────────────────────────────────────
    # 1. Separabilidad intra/inter-prompt: mide si la variabilidad semántica
    #    supera a la variabilidad estocástica del proceso de denoising.
    # 2. Dimensionalidad efectiva (PR): cuantifica cuántas dimensiones son
    #    realmente activas, con independencia de la dimensión nominal.
    # 3. Correlación latente↔imagen: valida que las distancias en z₀ preservan
    #    similitud perceptual medida sobre las imágenes decodificadas.
    # 4. Visualizaciones: una figura por modelo + comparativa cross-model.

    # 1. Matriz de distancias
    print("    [1/4] Matriz de distancias L2...")
    D_mat = compute_distance_matrix(Z_base)
    intra, inter = intra_inter_distances(D_mat, mf_base)
    sep = separability_ratio(intra, inter)
    ks_stat, ks_pval = ks_2samp(intra, inter)
    print(f"    intra μ={intra.mean():.2f}  inter μ={inter.mean():.2f}  sep={sep:.2f}  "
          f"KS={ks_stat:.3f} p={ks_pval:.2e}")

    # 2. Dimensionalidad efectiva
    print("    [2/4] Dimensionalidad efectiva...")
    er = effective_rank(Z_base)
    print(f"    PR={er['participation_ratio']:.1f}  "
          f"dims@80%={er['dims_80pct_variance']}  "
          f"dims@90%={er['dims_90pct_variance']}  "
          f"dims@95%={er['dims_95pct_variance']}")

    # 3. Correlación latente ↔ imagen
    print("    [3/4] Correlación latente ↔ imagen (muestreo de pares)...")
    corr = compute_latent_image_correlation(Z_base, mf_base, images_dir, max_pairs=500)
    print(f"    r(dist, MSE)={corr['pearson_latent_vs_mse']:.3f}  "
          f"ρ={corr['spearman_latent_vs_mse']:.3f}  "
          f"r(dist, -SSIM)={corr['pearson_latent_vs_neg_ssim']:.3f}")

    # 4. Plots
    print("    [4/4] Generando plots...")
    plot_distancias(D_mat, mf_base, intra, inter, model)
    plot_correlacion(corr, model)

    corr_clean = {k: v for k, v in corr.items() if not k.startswith("_")}
    return {
        "model": model,
        "latent_dim": int(D),
        "n_samples_base": int(N),
        "intra_mean": float(intra.mean()),
        "intra_std": float(intra.std()),
        "inter_mean": float(inter.mean()),
        "inter_std": float(inter.std()),
        "separability_ratio": sep,
        "ks_statistic_intra_vs_inter": float(ks_stat),
        "ks_pvalue_intra_vs_inter": float(ks_pval),
        "effective_rank": er,
        "correlation": corr_clean,
        # para plots cross-model (eliminados antes de JSON final)
        "_intra_dists": intra,
        "_inter_dists": inter,
    }


def main():
    print("=" * 60)
    print("  Fase 3 — Análisis geométrico del espacio latente")
    print("=" * 60)
    np.random.seed(42)

    all_results = {}
    for model in MODELS:
        res = run_model(model)
        if res is not None:
            all_results[model] = res

    if not all_results:
        print("\n[ERROR] No se procesó ningún modelo.")
        return

    print(f"\n  Generando comparativa cross-model...")
    plot_cross_model(all_results)

    # Serializar (sin arrays numpy)
    summary = {
        "fase": 3,
        "subtarea": "geometrica",
        "fecha": datetime.now().isoformat(),
        "modelos": {m: {k: v for k, v in r.items() if not k.startswith("_")}
                    for m, r in all_results.items()},
    }
    out = RESULTS_DIR / "geometrica_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Resumen guardado: {out.name}")
    print("  Figuras:")
    for m in all_results:
        print(f"    geo_{m}_distancias.png   geo_{m}_correlacion.png")
    print("    geo_cross_model.png")
    print("\nFase 3 completada.\n")


if __name__ == "__main__":
    main()
