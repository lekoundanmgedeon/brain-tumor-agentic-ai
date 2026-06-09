"""
Service de visualisation — Génération des figures pour Streamlit.
Prépare les affichages : image originale, masque, heatmap, métriques.
"""

import logging
import io
from typing import Optional

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Backend non-interactif (nécessaire pour Streamlit)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logger = logging.getLogger(__name__)


def load_image_for_display(image_path: str) -> Optional[Image.Image]:
    """
    Charge une image PIL pour l'affichage Streamlit.

    Args:
        image_path: Chemin vers l'image

    Returns:
        Image PIL ou None en cas d'erreur
    """
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"[Visualization] Impossible de charger l'image : {e}")
        return None


def create_analysis_figure(
    original_path: str,
    mask_path: Optional[str] = None,
    heatmap_path: Optional[str] = None,
    tumor_detected: bool = False,
    tumor_type: str = "unknown",
    confidence: float = 0.0,
) -> bytes:
    """
    Crée une figure matplotlib composite montrant :
    - L'image originale
    - Le masque de segmentation (si disponible)
    - La heatmap (si disponible)

    Args:
        original_path: Chemin vers l'image IRM originale
        mask_path: Chemin vers le masque de segmentation
        heatmap_path: Chemin vers la heatmap
        tumor_detected: Tumeur détectée ?
        tumor_type: Type de tumeur suspecté
        confidence: Score de confiance

    Returns:
        Bytes PNG de la figure matplotlib
    """
    # Déterminer le nombre de panneaux
    panels = ["Original"]
    if mask_path:
        panels.append("Masque")
    if heatmap_path:
        panels.append("Heatmap")

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    fig.patch.set_facecolor("#0e1117")  # Fond sombre (style Streamlit)

    # --- Panneau 1 : Image originale ---
    try:
        orig_img = Image.open(original_path).convert("RGB")
        axes[0].imshow(orig_img)
        axes[0].set_title("Image IRM Originale", color="white", fontsize=11, pad=8)
        axes[0].axis("off")
    except Exception as e:
        axes[0].text(0.5, 0.5, f"Erreur\n{e}", ha="center", va="center", color="red")
        axes[0].set_title("Image IRM Originale", color="white")

    # --- Panneau 2 : Masque de segmentation ---
    idx = 1
    if mask_path and idx < n_panels:
        try:
            mask_img = Image.open(mask_path).convert("L")
            axes[idx].imshow(mask_img, cmap="hot")
            axes[idx].set_title("Masque de Segmentation", color="white", fontsize=11, pad=8)
            axes[idx].axis("off")
            idx += 1
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Erreur\n{e}", ha="center", va="center", color="red")
            idx += 1

    # --- Panneau 3 : Heatmap ---
    if heatmap_path and idx < n_panels:
        try:
            heat_img = Image.open(heatmap_path).convert("RGB")
            axes[idx].imshow(heat_img)
            axes[idx].set_title("Heatmap d'Attention", color="white", fontsize=11, pad=8)
            axes[idx].axis("off")
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Erreur\n{e}", ha="center", va="center", color="red")

    # --- Titre global ---
    status_color = "#ff4b4b" if tumor_detected else "#21c55d"
    status_text = (
        f"⚠️ Anomalie Suspectée : {tumor_type.replace('_', ' ').title()} "
        f"(Confiance : {confidence:.1%})"
        if tumor_detected
        else f"✅ Aucune Anomalie Détectée (Confiance : {confidence:.1%})"
    )
    fig.suptitle(status_text, color=status_color, fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()

    # Convertir en bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()


def create_metrics_chart(
    confidence: float,
    quality_score: float,
    tumor_area: Optional[float] = None,
) -> bytes:
    """
    Crée un graphique en barres des métriques clés.

    Args:
        confidence: Score de confiance [0, 1]
        quality_score: Score de qualité de l'image [0, 1]
        tumor_area: Surface tumorale en mm² (optionnel)

    Returns:
        Bytes PNG du graphique
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1a2e")

    metrics = {
        "Confiance\nModèle": confidence,
        "Qualité\nImage": quality_score,
    }

    colors = []
    for val in metrics.values():
        if val >= 0.75:
            colors.append("#21c55d")
        elif val >= 0.4:
            colors.append("#f59e0b")
        else:
            colors.append("#ff4b4b")

    bars = ax.bar(list(metrics.keys()), list(metrics.values()), color=colors, width=0.5)

    # Ajouter les valeurs sur les barres
    for bar, val in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color="white")
    ax.set_title("Métriques d'Analyse", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()
