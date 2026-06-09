"""
Générateur de dataset IRM synthétique RÉALISTE.

Chaque classe est générée avec les vraies caractéristiques radiologiques :
  - glioma        : hyperintensité T2, contours irréguliers, œdème péritumoral
  - meningioma    : masse extra-axiale bien définie, homogène
  - pituitary_tumor : masse sellaire centrale, petite, bien délimitée
  - no_tumor      : anatomie normale, symétrie, pas d'anomalie

Ce dataset est utilisé UNIQUEMENT pour valider le pipeline technique.
Il ne remplace pas un vrai dataset médical (Kaggle, TCGA, BraTS...).
"""

import os
import numpy as np
from PIL import Image, ImageFilter
import random

SIZE = 224
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]

# Nombre d'images par classe par split
N_TRAIN = 300
N_VAL   = 75


# ─── Primitives anatomiques ──────────────────────────────────

def make_grid(size):
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    return x, y, cx, cy


def ellipse(x, y, cx, cy, rx, ry):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2


def brain_base(size=SIZE, rng=None) -> np.ndarray:
    """
    Crée une anatomie cérébrale de base réaliste en niveaux de gris [0,1].
    Inclut : crâne, LCR, matière grise, matière blanche, ventricules.
    """
    if rng is None:
        rng = np.random.default_rng()

    arr = np.zeros((size, size), dtype=np.float32)
    x, y, cx, cy = make_grid(size)

    # Légère asymétrie aléatoire (réalisme)
    ox = rng.integers(-4, 5)
    oy = rng.integers(-4, 5)

    # Crâne (hypersignal T1)
    skull_out = ellipse(x, y, cx + ox, cy + oy, size * 0.46, size * 0.43)
    skull_in  = ellipse(x, y, cx + ox, cy + oy, size * 0.41, size * 0.38)
    arr[(skull_out <= 1.0) & (skull_in > 1.0)] = 0.80 + rng.uniform(-0.05, 0.05)

    # LCR — espace sous-arachnoïdien (hyposignal T1)
    csf = ellipse(x, y, cx + ox, cy + oy, size * 0.40, size * 0.37)
    arr[csf <= 1.0] = 0.15

    # Matière grise (signal intermédiaire)
    gm = ellipse(x, y, cx + ox, cy + oy, size * 0.36, size * 0.33)
    arr[gm <= 1.0] = 0.45 + rng.uniform(-0.03, 0.03)

    # Matière blanche (hypersignal T1 par rapport à MG)
    wm = ellipse(x, y, cx + ox, cy + oy, size * 0.26, size * 0.23)
    arr[wm <= 1.0] = 0.62 + rng.uniform(-0.03, 0.03)

    # Ventricules latéraux (hyposignal T1 — LCR)
    vl = size * 0.16
    for vx_off, vy_off in [(-vl * 0.55, -vl * 0.1), (vl * 0.55, -vl * 0.1)]:
        v = ellipse(x, y, cx + ox + vx_off, cy + oy + vy_off, vl * 0.55, vl * 0.42)
        arr[v <= 1.0] = 0.08

    # 3e ventricule (médian)
    v3 = ellipse(x, y, cx + ox, cy + oy - size * 0.02, size * 0.025, size * 0.06)
    arr[v3 <= 1.0] = 0.08

    # Tronc cérébral / cervelet (bas de l'image)
    bs = ellipse(x, y, cx + ox, cy + oy + size * 0.30, size * 0.12, size * 0.10)
    arr[bs <= 1.0] = 0.50

    # Bruit gaussien (grain IRM)
    noise = rng.normal(0, 0.025, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 1)

    return arr


def add_glioma(arr: np.ndarray, rng) -> np.ndarray:
    """
    Ajoute un gliome simulé.
    Caractéristiques T1 + injection gadolinium :
      - Masse intra-axiale (dans le parenchyme)
      - Contours IRRÉGULIERS (lobulés)
      - Nécrose centrale (hyposignal)
      - Prise de contraste en anneau (hypersignal périphérique)
      - Œdème péritumoral T2 étendu (hyposignal T1 / hypersignal T2)
      - Effet de masse (déviation ligne médiane)
    """
    result = arr.copy()
    size = arr.shape[0]
    x, y, cx, cy = make_grid(size)

    # Position préférentielle : lobes frontaux, temporaux, pariétaux
    locations = [
        (cx - size * 0.15, cy - size * 0.12),  # frontal gauche
        (cx + size * 0.15, cy - size * 0.10),  # frontal droit
        (cx - size * 0.18, cy + size * 0.05),  # temporal gauche
        (cx + size * 0.18, cy + size * 0.05),  # temporal droit
        (cx - size * 0.10, cy - size * 0.18),  # pariétal gauche
        (cx + size * 0.10, cy - size * 0.18),  # pariétal droit
    ]
    tx, ty = locations[rng.integers(0, len(locations))]

    # Rayon principal
    rx = size * rng.uniform(0.08, 0.14)
    ry = size * rng.uniform(0.07, 0.12)

    # 1. Œdème péritumoral étendu (T2 hyposignal T1 → valeur basse)
    edema_scale = rng.uniform(1.6, 2.2)
    edema = ellipse(x, y, tx, ty, rx * edema_scale, ry * edema_scale)
    result[edema <= 1.0] = np.where(
        result[edema <= 1.0] > 0.1,
        result[edema <= 1.0] * 0.70 + rng.normal(0, 0.02, result[edema <= 1.0].shape),
        result[edema <= 1.0]
    )

    # 2. Masse tumorale (contours irréguliers via perturbation angulaire)
    angles = np.linspace(0, 2 * np.pi, 360)
    # Perturbation radiale aléatoire (contours lobulés)
    radial_noise = 1.0 + 0.25 * np.sin(rng.uniform(3, 8) * angles + rng.uniform(0, np.pi))
    for i, angle in enumerate(angles):
        rr = radial_noise[i]
        mask = ellipse(x, y, tx, ty, rx * rr, ry * rr)
        thin_ring = (mask <= 1.0) & (ellipse(x, y, tx, ty, rx * rr * 0.85, ry * rr * 0.85) > 1.0)
        result[thin_ring] = 0.85 + rng.uniform(-0.05, 0.05)  # prise de contraste

    # 3. Nécrose centrale (hyposignal franc)
    necrosis = ellipse(x, y, tx + rng.uniform(-5, 5), ty + rng.uniform(-5, 5),
                       rx * 0.45, ry * 0.45)
    result[necrosis <= 1.0] = 0.12 + rng.uniform(-0.03, 0.03)

    return np.clip(result, 0, 1)


def add_meningioma(arr: np.ndarray, rng) -> np.ndarray:
    """
    Ajoute un méningiome simulé.
    Caractéristiques radiologiques :
      - Masse EXTRA-axiale (contre la dure-mère, sur le bord du cerveau)
      - Contours NETS, réguliers, bien définis
      - Signal homogène (isosignal T1)
      - Prise de contraste INTENSE et homogène après gadolinium
      - Queue durale (dural tail) — extension le long de la méningée
      - Pas d'œdème ou œdème limité
    """
    result = arr.copy()
    size = arr.shape[0]
    x, y, cx, cy = make_grid(size)

    # Position EXTRA-axiale : convexité, falx, base du crâne
    edge_dist = size * 0.35   # distance par rapport au centre
    angle = rng.uniform(0, 2 * np.pi)
    tx = cx + edge_dist * np.cos(angle)
    ty = cy + edge_dist * np.sin(angle) * 0.85   # légèrement aplati

    # Taille modérée, contours réguliers
    rx = size * rng.uniform(0.06, 0.11)
    ry = size * rng.uniform(0.05, 0.09)

    # 1. Queue durale (extension linéaire vers la dure-mère)
    tail_len = rx * 2.5
    tail_dir_x = np.sign(tx - cx)
    tail_dir_y = np.sign(ty - cy)
    for t in np.linspace(0, 1, 20):
        tail_x = tx + tail_dir_x * tail_len * t
        tail_y = ty + tail_dir_y * tail_len * t
        tail_r = rx * 0.18 * (1 - t * 0.5)
        tail_mask = ellipse(x, y, tail_x, tail_y, tail_r, tail_r * 0.6)
        result[tail_mask <= 1.0] = 0.78 + rng.uniform(-0.03, 0.03)

    # 2. Corps tumoral — homogène, bien défini
    tumor_mask = ellipse(x, y, tx, ty, rx, ry)
    result[tumor_mask <= 1.0] = 0.82 + rng.normal(0, 0.015, result[tumor_mask <= 1.0].shape)

    # 3. Interface cerveau (légère dépression du parenchyme adjacent)
    compression = ellipse(x, y, tx - np.sign(tx - cx) * rx * 0.7,
                          ty - np.sign(ty - cy) * ry * 0.7, rx * 0.4, ry * 0.4)
    result[compression <= 1.0] = np.clip(result[compression <= 1.0] * 0.88, 0, 1)

    return np.clip(result, 0, 1)


def add_pituitary_tumor(arr: np.ndarray, rng) -> np.ndarray:
    """
    Ajoute un adénome hypophysaire simulé.
    Caractéristiques radiologiques :
      - Localisation STRICTEMENT sellaire (centre-bas de l'image)
      - Petite taille (micro < 10mm, macro > 10mm)
      - Contours réguliers, bien définis
      - Extension suprasellaire possible (macroAdénome)
      - Tige pituitaire déviée si volumineux
    """
    result = arr.copy()
    size = arr.shape[0]
    x, y, cx, cy = make_grid(size)

    # Position sellaire (selle turcique — centre légèrement bas)
    tx = cx + rng.uniform(-size * 0.02, size * 0.02)
    ty = cy + size * 0.22   # bas de l'hypothalamus

    # Taille : micro (5-9mm) ou macro (10-30mm)
    is_macro = rng.random() > 0.4
    if is_macro:
        rx = size * rng.uniform(0.05, 0.09)
        ry = size * rng.uniform(0.04, 0.08)
    else:
        rx = size * rng.uniform(0.025, 0.045)
        ry = size * rng.uniform(0.022, 0.040)

    # 1. Masse sellaire (hypersignal post-contraste)
    tumor_mask = ellipse(x, y, tx, ty, rx, ry)
    result[tumor_mask <= 1.0] = 0.80 + rng.normal(0, 0.018, result[tumor_mask <= 1.0].shape)

    # 2. Extension suprasellaire (macroadénome — forme en bonnet de gendarme)
    if is_macro:
        supra_ry = ry * rng.uniform(0.6, 1.0)
        supra = ellipse(x, y, tx, ty - ry * 1.3, rx * 0.85, supra_ry)
        result[supra <= 1.0] = 0.77 + rng.normal(0, 0.018, result[supra <= 1.0].shape)

        # Compression chiasmatique (légère)
        chiasm = ellipse(x, y, tx, ty - ry * 2.2, size * 0.07, size * 0.018)
        result[chiasm <= 1.0] = np.clip(result[chiasm <= 1.0] * 0.75, 0, 1)

        # Déviation de la tige pituitaire
        stalk_x = tx + rng.uniform(-size * 0.04, size * 0.04)
        stalk = ellipse(x, y, stalk_x, ty - ry * 0.9, size * 0.012, size * 0.04)
        result[stalk <= 1.0] = 0.72

    return np.clip(result, 0, 1)


def post_process(arr: np.ndarray, rng) -> np.ndarray:
    """
    Post-traitement pour augmenter le réalisme :
    - Rotation aléatoire
    - Légère variation de contraste
    - Flou gaussien léger (PSF IRM)
    - Artefacts de mouvement subtils
    """
    from PIL import Image, ImageFilter, ImageEnhance

    # Convertir en image PIL
    img_uint8 = (arr * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8, mode='L')

    # Rotation aléatoire (±15°)
    angle = rng.uniform(-15, 15)
    img = img.rotate(angle, fillcolor=0, resample=Image.BILINEAR)

    # Flou PSF (Point Spread Function)
    sigma = rng.uniform(0.3, 0.8)
    img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    # Variation de contraste légère
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(rng.uniform(0.85, 1.15))

    # Retour en numpy
    return np.array(img, dtype=np.float32) / 255.0


def generate_sample(class_name: str, rng, size=SIZE) -> np.ndarray:
    """Génère une image IRM synthétique pour une classe donnée."""
    base = brain_base(size, rng)
    if class_name == "glioma":
        arr = add_glioma(base, rng)
    elif class_name == "meningioma":
        arr = add_meningioma(base, rng)
    elif class_name == "pituitary_tumor":
        arr = add_pituitary_tumor(base, rng)
    else:  # no_tumor
        arr = base.copy()
        # Légères variations normales (gyri, asymétrie physiologique)
        x, y, cx, cy = make_grid(size)
        for _ in range(rng.integers(2, 5)):
            gx = rng.uniform(cx - size*0.25, cx + size*0.25)
            gy = rng.uniform(cy - size*0.25, cy + size*0.25)
            gr = rng.uniform(size*0.02, size*0.04)
            gyrus = ellipse(x, y, gx, gy, gr, gr * 0.7)
            arr[gyrus <= 1.0] = np.clip(arr[gyrus <= 1.0] + rng.uniform(0.03, 0.07), 0, 1)

    return post_process(arr, rng)


# ─── Génération du dataset ────────────────────────────────────

def generate_dataset(output_dir: str = "data/synthetic_mri",
                     n_train: int = N_TRAIN,
                     n_val: int = N_VAL,
                     seed: int = 42):
    """
    Génère le dataset complet et le sauvegarde dans la structure
    attendue par torchvision.datasets.ImageFolder :

        output_dir/
          Training/
            glioma/           (n_train images)
            meningioma/
            no_tumor/
            pituitary_tumor/
          Testing/
            glioma/           (n_val images)
            meningioma/
            no_tumor/
            pituitary_tumor/
    """
    import os
    from tqdm import tqdm

    rng_master = np.random.default_rng(seed)

    total = (n_train + n_val) * len(CLASSES)
    print(f"\nGénération du dataset synthétique réaliste")
    print(f"  {n_train} images train + {n_val} images val par classe")
    print(f"  Total : {total} images IRM\n")

    generated = 0
    for split, n_imgs in [("Training", n_train), ("Testing", n_val)]:
        for cls in CLASSES:
            folder = os.path.join(output_dir, split, cls)
            os.makedirs(folder, exist_ok=True)

            desc = f"{split}/{cls}"
            for i in range(n_imgs):
                rng = np.random.default_rng(rng_master.integers(0, 2**31))
                arr = generate_sample(cls, rng)

                # Sauvegarder en RGB (requis par ImageFolder + EfficientNet)
                img_uint8 = (arr * 255).astype(np.uint8)
                img = Image.fromarray(img_uint8, mode='L').convert('RGB')
                path = os.path.join(folder, f"{cls}_{i:04d}.png")
                img.save(path, optimize=True)

                generated += 1
                if generated % 50 == 0:
                    print(f"  [{generated:4d}/{total}] {desc}_{i:04d}.png")

    print(f"\n✅ Dataset généré dans : {output_dir}/")
    print(f"   {total} images au total")
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/synthetic_mri")
    parser.add_argument("--n_train", type=int, default=N_TRAIN)
    parser.add_argument("--n_val",   type=int, default=N_VAL)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.output_dir, args.n_train, args.n_val, args.seed)
