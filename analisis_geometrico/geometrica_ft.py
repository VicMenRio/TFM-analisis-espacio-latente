"""
Fase 3 — Análisis del impacto del fine-tuning en el espacio latente.

Modos de uso
─────────────
  snapshot   Calcula y guarda estadísticas pretrain (PCA, centroide, direcciones
             semánticas). Ejecutar antes del fine-tuning.

  compare    Carga el snapshot pretrain + latents finetune y calcula métricas
             de deriva (desplazamiento geométrico, retención de varianza PCA,
             estabilidad semántica). Ejecutar después del fine-tuning.

Uso
─────────────
  python analisis.py snapshot
  python analisis.py compare

─── Formato de datos ────────────────────────────────────────────────────────
  corpus/latents/{model}/
    latents.pt      Tensor [N, 1, C, H, W]
    manifest.json   lista de { id, category, prompt, seed, idx }

  Para compare, los latents finetune deben estar en:
    ajuste_fino/latents_post_ft/{model}/
      latents.pt
      manifest.json

─── Salidas ─────────────────────────────────────────────────────────────────
  analisis_geometrico/resultados/
    snapshot_{model}.pkl          estadísticas pretrain serializadas
    snapshot_{model}_report.json  resumen legible
    {mode}_{model}_pca.png
    {mode}_{model}_semantic.png
    cross_model_comparison.png    (solo en compare)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent
LATENTS_DIR = ROOT / "corpus" / "latents"
LATENTS_FT_DIR = ROOT / "ajuste_fino" / "latents_post_ft"
RESULTS_DIR = ROOT / "analisis_geometrico" / "resultados"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["sd15", "sd21", "sdxl"]
PCA_COMPONENTS = 50


# ─── Carga ───────────────────────────────────────────────────────────────────

def load_model_data(base_dir: Path, model: str) -> tuple[np.ndarray, list[dict]]:
    """
    Devuelve (Z, manifest) donde Z tiene shape [N, D] (aplanado).
    """
    # El tensor puede tener shape [N,1,C,H,W] (guardado con batch dim) o [N,C,H,W].
    # Se aplana a [N, D] para que todos los análisis trabajen con vectores 1D,
    # independientemente de la resolución original del modelo.
    pt_path = base_dir / model / "latents.pt"
    mf_path = base_dir / model / "manifest.json"
    if not pt_path.exists():
        raise FileNotFoundError(pt_path)
    tensor = torch.load(pt_path, map_location="cpu", weights_only=False).float()
    # [N, 1, C, H, W] o [N, C, H, W]
    if tensor.ndim == 5:
        tensor = tensor.squeeze(1)            # → [N, C, H, W]
    N = tensor.shape[0]
    Z = tensor.numpy().reshape(N, -1)         # → [N, D]
    with open(mf_path) as f:
        manifest = json.load(f)
    assert len(manifest) == N, f"{model}: manifest len {len(manifest)} ≠ tensor rows {N}"
    return Z, manifest


def split_by_category(Z: np.ndarray, manifest: list[dict]):
    base_mask = np.array([e["category"] == "base" for e in manifest])
    attr_mask = ~base_mask
    Z_base = Z[base_mask]
    Z_attr = Z[attr_mask]
    manifest_base = [e for e in manifest if e["category"] == "base"]
    manifest_attr = [e for e in manifest if e["category"] != "base"]
    return Z_base, manifest_base, Z_attr, manifest_attr


def build_group_means(Z: np.ndarray, manifest: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    """
    Devuelve { group: { value: mean_vector } } para los prompts de atributo.
    """
    from collections import defaultdict
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for i, e in enumerate(manifest):
        if e["category"] == "attribute":
            acc[e.get("group", e["id"])][e.get("attribute_value", "unknown")].append(Z[i])
    result = {}
    for group, vals in acc.items():
        result[group] = {v: np.stack(vecs).mean(axis=0) for v, vecs in vals.items()}
    return result



# ─── MODO SNAPSHOT ───────────────────────────────────────────────────────────

def run_snapshot(model: str) -> dict:
    # Modo snapshot: captura el estado del espacio latente ANTES del fine-tuning.
    # Calcula y serializa las estadísticas geométricas y semánticas del pretrain
    # (PCA, centroide, normas, distribución de distancias, direcciones semánticas)
    # en un archivo .pkl que será la referencia para el modo compare posterior.
    # Ejecutar: python analisis.py snapshot
    print(f"\n  [{model.upper()}] Cargando datos...")
    Z, manifest = load_model_data(LATENTS_DIR, model)
    Z_base, mf_base, Z_attr, mf_attr = split_by_category(Z, manifest)
    D = Z.shape[1]
    print(f"    shape total: {Z.shape}  base: {Z_base.shape}  attr: {Z_attr.shape}")

    # PCA sobre prompts base
    n_comp = min(PCA_COMPONENTS, Z_base.shape[0], Z_base.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full")
    pca.fit(Z_base)
    Z_base_proj = pca.transform(Z_base)   # [N_base, n_comp]

    # Centroide y norma media
    centroid = Z_base.mean(axis=0)        # [D]
    norms = np.linalg.norm(Z_base, axis=1)

    # Distribución de distancias intra-conjunto (muestreo)
    n_sample = min(len(Z_base), 200)
    idx = np.random.choice(len(Z_base), n_sample, replace=False)
    Z_sub = Z_base[idx]
    diff = Z_sub[:, None, :] - Z_sub[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)[np.triu_indices(n_sample, k=1)]

    # Direcciones semánticas (prompts de atributo)
    groups = build_group_means(Z, manifest)
    semantic_dirs: dict[str, np.ndarray] = {}
    for grp, vals in groups.items():
        if len(vals) == 2:
            v1, v2 = sorted(vals.keys())
            semantic_dirs[grp] = vals[v1] - vals[v2]    # dirección normalizable

    # Serializar snapshot
    snapshot = {
        "model": model,
        "fecha": datetime.now().isoformat(),
        "latent_dim": int(D),
        "n_samples_base": int(Z_base.shape[0]),
        "n_samples_attr": int(Z_attr.shape[0]),
        "pca": pca,
        "centroid": centroid,
        "norms_mean": float(norms.mean()),
        "norms_std": float(norms.std()),
        "dist_sample": dists,
        "semantic_dirs": semantic_dirs,
        "Z_base_proj": Z_base_proj,
        "variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    snap_path = RESULTS_DIR / f"snapshot_{model}.pkl"
    with open(snap_path, "wb") as f:
        pickle.dump(snapshot, f)
    print(f"    Snapshot guardado: {snap_path.name}")

    # Reporte JSON (sin arrays grandes)
    report = {
        "model": model,
        "fecha": snapshot["fecha"],
        "latent_dim": int(D),
        "n_samples_base": int(Z_base.shape[0]),
        "n_comp_pca": n_comp,
        "var_explained_top10": float(pca.explained_variance_ratio_[:10].sum()),
        "var_explained_top50": float(pca.explained_variance_ratio_.sum()),
        "centroid_norm": float(np.linalg.norm(centroid)),
        "norms_mean": float(norms.mean()),
        "norms_std": float(norms.std()),
        "dist_mean": float(dists.mean()),
        "dist_std": float(dists.std()),
        "semantic_groups": list(semantic_dirs.keys()),
        "semantic_dir_norms": {g: float(np.linalg.norm(d)) for g, d in semantic_dirs.items()},
    }
    rpt_path = RESULTS_DIR / f"snapshot_{model}_report.json"
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)

    # Plot snapshot
    _plot_snapshot(snapshot, model)

    return report


def _plot_snapshot(snap: dict, model: str):
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    fig.suptitle(f"{model.upper()} — Snapshot pretrain del espacio latente", fontsize=13)

    # Varianza explicada acumulada
    ax1 = fig.add_subplot(gs[0])
    k = min(30, len(snap["variance_ratio"]))
    var_cum = np.cumsum(snap["variance_ratio"])[:k]
    ax1.plot(range(1, k + 1), var_cum, "o-", color="#4C72B0", markersize=4)
    ax1.axhline(0.8, color="orange", linestyle="--", alpha=0.6, label="80%")
    ax1.axhline(0.9, color="red", linestyle=":", alpha=0.6, label="90%")
    ax1.set_xlabel("Nº componentes")
    ax1.set_ylabel("Varianza explicada acumulada")
    ax1.set_title("PCA — Varianza acumulada")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Proyección PC1 vs PC2
    ax2 = fig.add_subplot(gs[1])
    proj = snap["Z_base_proj"]
    ax2.scatter(proj[:, 0], proj[:, 1], alpha=0.6, s=20, color="#4C72B0")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Proyección base: PC1 vs PC2")
    ax2.grid(alpha=0.3)

    # Normas de direcciones semánticas
    ax3 = fig.add_subplot(gs[2])
    dirs = snap["semantic_dirs"]
    if dirs:
        names = list(dirs.keys())
        norms = [float(np.linalg.norm(dirs[n])) for n in names]
        short = [n.replace("color_", "col:").replace("light_", "luz:").replace("style_", "est:") for n in names]
        ax3.barh(range(len(names)), norms, color="#4C72B0", alpha=0.8)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(short, fontsize=8)
        ax3.set_xlabel("||dirección semántica||")
        ax3.set_title("Magnitud de direcciones semánticas\n(pretrain)")
        ax3.grid(axis="x", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Sin datos de atributo", ha="center", va="center")
        ax3.axis("off")

    plt.tight_layout()
    out = RESULTS_DIR / f"snapshot_{model}_pca.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


# ─── MODO COMPARE ────────────────────────────────────────────────────────────

def run_compare(model: str) -> dict | None:
    snap_path = RESULTS_DIR / f"snapshot_{model}.pkl"
    if not snap_path.exists():
        print(f"  [{model.upper()}] No hay snapshot pretrain. Ejecuta primero: python analisis.py snapshot")
        return None

    ft_dir = LATENTS_FT_DIR
    if not (ft_dir / model / "latents.pt").exists():
        print(f"  [{model.upper()}] No se encuentran latents finetune en {ft_dir / model}")
        return None

    print(f"\n  [{model.upper()}] Cargando snapshot y latents finetune...")
    with open(snap_path, "rb") as f:
        snap = pickle.load(f)

    Z_ft, mf_ft = load_model_data(ft_dir, model)
    Z_ft_base, mf_ft_base, _, _ = split_by_category(Z_ft, mf_ft)

    pca_pre: PCA = snap["pca"]
    Z_pre_base_proj = snap["Z_base_proj"]

    # Necesitamos el tensor pretrain para métricas emparejadas
    Z_pre, mf_pre = load_model_data(LATENTS_DIR, model)
    Z_pre_base, mf_pre_base, _, _ = split_by_category(Z_pre, mf_pre)

    # Alinear por prompt_id + seed
    pre_index = {(e["id"], e["seed"]): i for i, e in enumerate(mf_pre_base)}
    ft_index  = {(e["id"], e["seed"]): i for i, e in enumerate(mf_ft_base)}
    common_keys = sorted(set(pre_index) & set(ft_index))
    if len(common_keys) == 0:
        print(f"  [{model.upper()}] No hay pares (prompt_id, seed) comunes entre pretrain y finetune.")
        return None

    pre_rows = np.array([pre_index[k] for k in common_keys])
    ft_rows  = np.array([ft_index[k]  for k in common_keys])
    Z_pre_paired = Z_pre_base[pre_rows]   # [N_paired, D]
    Z_ft_paired  = Z_ft_base[ft_rows]     # [N_paired, D]
    print(f"    Pares alineados: {len(common_keys)}")

    # ── 1. Geometría ──────────────────────────────────────────────────────
    delta = Z_ft_paired - Z_pre_paired
    disp = np.linalg.norm(delta, axis=1)
    centroid_pre = Z_pre_paired.mean(axis=0)
    centroid_ft  = Z_ft_paired.mean(axis=0)
    centroid_shift = float(np.linalg.norm(centroid_ft - centroid_pre))
    centroid_rel   = centroid_shift / (float(np.linalg.norm(centroid_pre)) + 1e-8)

    norms_pre = np.linalg.norm(Z_pre_paired, axis=1, keepdims=True) + 1e-8
    norms_ft  = np.linalg.norm(Z_ft_paired,  axis=1, keepdims=True) + 1e-8
    cos_sim_paired = float((Z_pre_paired / norms_pre * Z_ft_paired / norms_ft).sum(axis=1).mean())

    n_sub = min(len(Z_pre_paired), 150)
    idx = np.random.choice(len(Z_pre_paired), n_sub, replace=False)
    diff_pre = Z_pre_paired[idx][:, None] - Z_pre_paired[idx][None, :]
    diff_ft  = Z_ft_paired[idx][:, None]  - Z_ft_paired[idx][None, :]
    dists_pre = np.linalg.norm(diff_pre, axis=-1)[np.triu_indices(n_sub, k=1)]
    dists_ft  = np.linalg.norm(diff_ft,  axis=-1)[np.triu_indices(n_sub, k=1)]
    ks_stat, ks_pval = ks_2samp(dists_pre, dists_ft)

    # ── 2. PCA ────────────────────────────────────────────────────────────
    n_comp = pca_pre.n_components_

    pca_ft = PCA(n_components=n_comp, svd_solver="full")
    pca_ft.fit(Z_ft_base)

    Z_ft_proj_on_pre = pca_pre.transform(Z_ft_base)
    post_centered = Z_ft_base - pca_pre.mean_
    var_total  = float(np.var(post_centered, axis=0).sum())
    var_on_pre = float(np.var(Z_ft_proj_on_pre, axis=0).sum())
    var_retention = var_on_pre / (var_total + 1e-8)

    # ── 3. Semántica ──────────────────────────────────────────────────────
    groups_ft = build_group_means(Z_ft, mf_ft)
    dirs_pre = snap["semantic_dirs"]
    semantic_results = {}
    for grp, dir_pre in dirs_pre.items():
        if grp not in groups_ft:
            continue
        ft_vals = groups_ft[grp]
        if len(ft_vals) == 2:
            v1, v2 = sorted(ft_vals.keys())
            dir_ft = ft_vals[v1] - ft_vals[v2]
            n_pre = np.linalg.norm(dir_pre) + 1e-8
            n_ft  = np.linalg.norm(dir_ft) + 1e-8
            cos = float(np.dot(dir_pre / n_pre, dir_ft / n_ft))
            semantic_results[grp] = {
                "cosine_similarity": cos,
                "norm_pre": float(n_pre),
                "norm_ft": float(n_ft),
                "norm_ratio": float(n_ft / n_pre),
            }

    results = {
        "model": model,
        "fecha": datetime.now().isoformat(),
        "n_paired": len(common_keys),
        "geometry": {
            "centroid_shift_L2": centroid_shift,
            "centroid_shift_relative": centroid_rel,
            "mean_paired_displacement": float(disp.mean()),
            "std_paired_displacement": float(disp.std()),
            "mean_cosine_similarity_paired": cos_sim_paired,
            "pairwise_dist_mean_pre": float(dists_pre.mean()),
            "pairwise_dist_mean_ft": float(dists_ft.mean()),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
        },
        "pca": {
            "variance_retention_on_pre_basis": var_retention,
            "var_explained_pre": pca_pre.explained_variance_ratio_[:20].tolist(),
            "var_explained_ft": pca_ft.explained_variance_ratio_[:20].tolist(),
        },
        "semantic": semantic_results,
        # arrays para plots (se limpian antes del JSON)
        "_dists_pre": dists_pre,
        "_dists_ft": dists_ft,
        "_pre_proj": Z_pre_paired[:, :2] if Z_pre_paired.shape[1] >= 2 else None,
        "_ft_proj": Z_ft_paired[:, :2] if Z_ft_paired.shape[1] >= 2 else None,
        "_pre_proj_pca": pca_pre.transform(Z_pre_paired),
        "_ft_proj_pca": pca_pre.transform(Z_ft_paired),
        "_var_pre": pca_pre.explained_variance_ratio_,
        "_var_ft": pca_ft.explained_variance_ratio_,
    }

    _plot_compare(results, model)
    return results


def _plot_compare(res: dict, model: str):
    geo = res["geometry"]
    pca_r = res["pca"]
    sem = res["semantic"]

    # ── Plot geométrico ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{model.upper()} — Deriva geométrica (pre vs post fine-tuning)", fontsize=13)

    ax = axes[0]
    bins = 40
    ax.hist(res["_dists_pre"], bins=bins, alpha=0.6, label="Pre", color="#4C72B0")
    ax.hist(res["_dists_ft"],  bins=bins, alpha=0.6, label="Post FT", color="#DD8452")
    ax.set_xlabel("Distancia euclídea intra-conjunto")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de distancias")
    ax.legend()
    ax.text(0.97, 0.97, f"KS={geo['ks_statistic']:.3f}  p={geo['ks_pvalue']:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.2))
    ax.grid(alpha=0.3)

    ax = axes[1]
    proj_pre = res["_pre_proj_pca"]
    proj_ft  = res["_ft_proj_pca"]
    ax.scatter(proj_pre[:, 0], proj_pre[:, 1], alpha=0.5, s=20, label="Pre", color="#4C72B0")
    ax.scatter(proj_ft[:, 0],  proj_ft[:, 1],  alpha=0.5, s=20, label="Post FT", color="#DD8452", marker="^")
    c0 = proj_pre.mean(axis=0)
    c1 = proj_ft.mean(axis=0)
    ax.annotate("", xy=c1[:2], xytext=c0[:2],
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.set_xlabel("PC1 (base pre)"); ax.set_ylabel("PC2 (base pre)")
    ax.set_title("PC1 vs PC2")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    k = min(20, len(pca_r["var_explained_pre"]))
    x = range(1, k + 1)
    ax.plot(x, np.cumsum(pca_r["var_explained_pre"])[:k], "o-", label="Pre", color="#4C72B0", markersize=4)
    ax.plot(x, np.cumsum(pca_r["var_explained_ft"])[:k], "s--", label="Post FT", color="#DD8452", markersize=4)
    ax.set_xlabel("Nº componentes"); ax.set_ylabel("Var. acumulada")
    ax.set_title(f"PCA — varianza acumulada\nRetención={pca_r['variance_retention_on_pre_basis']:.3f}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"compare_{model}_pca.png", dpi=120, bbox_inches="tight")
    plt.close()

    if not sem:
        return

    # ── Plot semántico ──
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.45 * len(sem) + 2)))
    fig.suptitle(f"{model.upper()} — Estabilidad de direcciones semánticas", fontsize=13)
    groups = list(sem.keys())
    cos_vals = [sem[g]["cosine_similarity"] for g in groups]
    ratio_vals = [sem[g]["norm_ratio"] for g in groups]
    short = [g.replace("color_", "col:").replace("light_", "luz:").replace("style_", "est:") for g in groups]
    y = range(len(groups))

    ax = axes[0]
    colors = ["#4CAF50" if c > 0.9 else ("#FF9800" if c > 0.7 else "#F44336") for c in cos_vals]
    ax.barh(list(y), cos_vals, color=colors, alpha=0.85)
    ax.set_yticks(list(y)); ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Similitud coseno (dir_pre · dir_post)")
    ax.set_title("Estabilidad de dirección\n(1.0 = sin cambio)"); ax.set_xlim(max(0, min(cos_vals) - 0.05), 1.05)
    ax.axvline(1.0, color="green", ls="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    ax.barh(list(y), ratio_vals, color="#4C72B0", alpha=0.8)
    ax.set_yticks(list(y)); ax.set_yticklabels(short, fontsize=9)
    ax.axvline(1.0, color="black", ls="--", alpha=0.7)
    ax.set_xlabel("||dir_post|| / ||dir_pre||")
    ax.set_title("Magnitud relativa\n(1.0 = misma magnitud)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"compare_{model}_semantic.png", dpi=120, bbox_inches="tight")
    plt.close()


def _plot_cross_model(all_compare: dict):
    models = list(all_compare.keys())
    if len(models) < 2:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Fase 3 — Comparativa cross-model: impacto del fine-tuning", fontsize=14)
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    geo_metrics = [
        ("Δ centroide (rel.)", "centroid_shift_relative"),
        ("Despl. emparejado μ", "mean_paired_displacement"),
        ("KS statistic", "ks_statistic"),
    ]
    for ax, (label, key) in zip(axes[0], geo_metrics):
        vals = [all_compare[m]["geometry"][key] for m in models]
        bars = ax.bar(models, vals, color=colors[:len(models)], alpha=0.85)
        ax.set_title(label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    pca_metrics = [
        ("Retención de varianza", "variance_retention_on_pre_basis"),
    ]
    for ax, (label, key) in zip(axes[1][:1], pca_metrics):
        vals = [all_compare[m]["pca"][key] for m in models]
        bars = ax.bar(models, vals, color=colors[:len(models)], alpha=0.85)
        ax.set_title(label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    ax = axes[1][2]
    sem_means = []
    for m in models:
        sems = all_compare[m].get("semantic", {})
        cos_vals = [v["cosine_similarity"] for v in sems.values()] if sems else []
        sem_means.append(float(np.mean(cos_vals)) if cos_vals else float("nan"))
    bars = ax.bar(models, sem_means, color=colors[:len(models)], alpha=0.85)
    ax.set_title("Estabilidad semántica media\n(cos sim dir pre/post)")
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, sem_means):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cross_model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  cross_model_comparison.png guardado.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["snapshot", "compare"],
                        help="snapshot: calcula estadísticas pretrain. compare: analiza deriva post fine-tuning.")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help=f"Modelos a procesar (default: {MODELS})")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Fase 3 — modo: {args.mode.upper()}")
    print("=" * 60)

    np.random.seed(42)

    if args.mode == "snapshot":
        reports = {}
        for model in args.models:
            try:
                rpt = run_snapshot(model)
                reports[model] = rpt
                print(f"    var top10={rpt['var_explained_top10']:.3f}  "
                      f"centroid_norm={rpt['centroid_norm']:.2f}  "
                      f"dist_mean={rpt['dist_mean']:.2f}")
            except FileNotFoundError as e:
                print(f"  [{model.upper()}] SKIP: {e}")

        summary = {"fase": 3, "modo": "snapshot", "fecha": datetime.now().isoformat(),
                   "modelos": reports}
        out = RESULTS_DIR / "snapshot_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Resumen: {out}")

        # Plot comparativo de varianza entre modelos en snapshot
        if len(reports) >= 2:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.suptitle("Snapshot pretrain — varianza explicada acumulada por modelo", fontsize=12)
            colors = {"sd15": "#4C72B0", "sd21": "#DD8452", "sdxl": "#55A868"}
            for model, rpt in reports.items():
                snap_path = RESULTS_DIR / f"snapshot_{model}.pkl"
                with open(snap_path, "rb") as f:
                    snap = pickle.load(f)
                k = min(30, len(snap["variance_ratio"]))
                ax.plot(range(1, k + 1), np.cumsum(snap["variance_ratio"])[:k],
                        "o-", label=model.upper(), color=colors.get(model), markersize=4)
            ax.axhline(0.8, color="gray", ls="--", alpha=0.5, label="80%")
            ax.set_xlabel("Nº componentes PCA"); ax.set_ylabel("Varianza acumulada")
            ax.legend(); ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / "snapshot_cross_model_pca.png", dpi=120, bbox_inches="tight")
            plt.close()
            print(f"  snapshot_cross_model_pca.png guardado.")

    elif args.mode == "compare":
        all_compare = {}
        for model in args.models:
            res = run_compare(model)
            if res is None:
                continue
            # Limpiar arrays antes de serializar
            res_clean = {k: v for k, v in res.items() if not k.startswith("_")}
            all_compare[model] = res_clean

            geo = res_clean["geometry"]
            pca_r = res_clean["pca"]
            print(f"    centroid_rel={geo['centroid_shift_relative']:.4f}  "
                  f"cos_sim={geo['mean_cosine_similarity_paired']:.4f}  "
                  f"KS={geo['ks_statistic']:.4f}  "
                  f"var_ret={pca_r['variance_retention_on_pre_basis']:.4f}")

        if all_compare:
            if len(all_compare) >= 2:
                _plot_cross_model(all_compare)
            out = RESULTS_DIR / "comparacion_summary.json"
            with open(out, "w") as f:
                json.dump({"fase": 3, "modo": "compare",
                           "fecha": datetime.now().isoformat(),
                           "modelos": all_compare}, f, indent=2)
            print(f"\n  Resumen: {out}")

    print("\nFase 3 completada.\n")


if __name__ == "__main__":
    main()
