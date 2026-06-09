# 🚀 Guide — Mode Production avec Gemini + Modèle Réel

> Ce guide explique comment passer du mode DEMO au mode PRODUCTION_READY
> avec un vrai modèle de classification et Google Gemini.

---

## Étape 1 — Prérequis

```bash
# Dans le dossier du projet
cd brain-tumor-agentic-ai

# Activer l'environnement virtuel
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

---

## Étape 2 — Installer PyTorch + google-genai

```bash
# PyTorch (CPU — suffisant pour l'inférence)
pip install torch torchvision

# SDK Gemini v2 (remplace google-generativeai déprécié)
pip install google-genai

# Optionnel : Mistral
pip install mistralai
```

---

## Étape 3 — Obtenir une clé Gemini

1. Allez sur **https://aistudio.google.com/**
2. Connectez-vous avec votre compte Google
3. Cliquez sur **"Get API Key"** → **"Create API key"**
4. Copiez la clé (commence par `AIza...`)

> La formule **gratuite** de Gemini (Free Tier) est suffisante pour tester :
> - 15 requêtes/minute
> - 1 million de tokens/jour
> - Modèles disponibles : `gemini-2.0-flash`, `gemini-1.5-flash`

---

## Étape 4 — Configurer le fichier .env

```bash
cp .env.example .env
```

Éditez `.env` :

```env
# ─── MODE PRODUCTION ──────────────────────────────
APP_MODE=PRODUCTION_READY
LLM_PROVIDER=gemini

# ─── GEMINI ───────────────────────────────────────
GEMINI_API_KEY=AIzaSy...votre-cle-ici
GEMINI_MODEL=gemini-2.0-flash

# ─── RAG (optionnel mais recommandé) ──────────────
PUBMED_EMAIL=votre@email.com
ENABLE_PUBMED=true
ENABLE_NCI_FALLBACK=true
```

---

## Étape 5 — Générer les poids du modèle

### Option A — Données synthétiques (validation du pipeline)

```bash
python models/download_weights.py
```

Génère un modèle EfficientNet-B0 entraîné sur 200 images aléatoires.
**Les prédictions ne sont pas médicalement valides** mais valident tout le pipeline.

### Option B — Dataset Kaggle réel (recommandé pour un projet sérieux)

**Dataset** : Brain Tumor MRI Dataset (~7000 images, 4 classes)
**URL** : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

```bash
# 1. Installer kaggle CLI
pip install kaggle

# 2. Placer votre kaggle.json dans ~/.kaggle/
#    (téléchargez-le depuis https://www.kaggle.com/settings → API)

# 3. Télécharger le dataset
kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/brain_tumor/

# 4. Entraîner le modèle (GPU recommandé, ~20 min sur CPU)
python models/train_real.py --data_dir data/brain_tumor/ --epochs 20
```

Structure attendue du dataset :
```
data/brain_tumor/
├── Training/
│   ├── glioma_tumor/        (~900 images)
│   ├── meningioma_tumor/    (~900 images)
│   ├── no_tumor/            (~500 images)
│   └── pituitary_tumor/     (~900 images)
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

---

## Étape 6 — Tester la connexion Gemini

```bash
python test_gemini_connection.py
```

Sortie attendue :
```
✅ Clé trouvée (longueur : 39 chars)
🔄 Test avec le modèle : gemini-2.0-flash (SDK google-genai v2)...
✅ Connexion Gemini réussie !
   Réponse : Je suis un assistant médical IA académique...
✅ Tout est prêt pour le mode PRODUCTION_READY + Gemini !
```

---

## Étape 7 — Lancer l'application

```bash
streamlit run app.py
```

En haut de la sidebar, vous verrez :
- `🔵 GEMINI` (si clé valide)
- `Mode : PRODUCTION_READY`

---

## Ce qui change en mode PRODUCTION_READY

| Composant | Mode DEMO | Mode PRODUCTION_READY |
|-----------|-----------|----------------------|
| **Détection** | Simulation statistique | EfficientNet-B0 réel |
| **Confiance** | Valeur aléatoire simulée | Probabilité softmax réelle |
| **Segmentation** | Ellipse géométrique | Grad-CAM (cartes d'activation) |
| **Heatmap** | Overlay rouge fixe | Heatmap JET colormap réelle |
| **Localisation** | Aléatoire | Centroïde Grad-CAM |
| **Rapport** | Template fixe | Généré par Gemini (LLM) |
| **RAG** | Base locale uniquement | PubMed + NCI + local |

---

## Modèles Gemini disponibles

| Modèle | Vitesse | Qualité | Usage |
|--------|---------|---------|-------|
| `gemini-2.0-flash` | ⚡⚡⚡ | ★★★★ | **Recommandé** — rapport de 2048 tokens en ~3s |
| `gemini-1.5-flash` | ⚡⚡⚡ | ★★★★ | Alternative stable |
| `gemini-1.5-pro`   | ⚡⚡   | ★★★★★ | Meilleure qualité, plus lent |

Changez le modèle dans `.env` : `GEMINI_MODEL=gemini-1.5-pro`

---

## Activer PubMed pour le RAG

```env
ENABLE_PUBMED=true
PUBMED_EMAIL=votre@email.com
```

L'API NCBI est gratuite et ne nécessite pas de clé.
L'email est requis pour les bonnes pratiques NCBI (rate limiting respectueux).

Résultats attendus : 3-5 articles PubMed récents sur le type de tumeur détecté.

---

## Intégrer MedSAM (segmentation médicale avancée)

```bash
# 1. Télécharger MedSAM
git clone https://github.com/bowang-lab/MedSAM.git
cd MedSAM && pip install -e .

# 2. Télécharger les poids (~375 MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# 3. Décommentez le bloc MedSAM dans services/segmentation.py
```

---

## Résumé des fichiers modifiés pour la production

```
services/
├── inference.py      ← EfficientNet-B0 + Grad-CAM (REMPLACÉ)
├── segmentation.py   ← Masque depuis Grad-CAM (REMPLACÉ)
└── llm_client.py     ← SDK google-genai v2 (REMPLACÉ)

models/
├── classifier.py     ← Architecture EfficientNet-B0 (NOUVEAU)
├── download_weights.py ← Génération poids synthétiques (NOUVEAU)
└── classifier_weights.pth ← Poids entraînés (GÉNÉRÉ)

config/settings.py    ← GEMINI_MODEL configurable (MIS À JOUR)
.env.example          ← Variables production complètes (MIS À JOUR)
requirements.txt      ← google-genai + torch (MIS À JOUR)
test_gemini_connection.py ← Test rapide API (NOUVEAU)
PRODUCTION_GUIDE.md   ← Ce guide (NOUVEAU)
```

---

## ⚠️ Rappel Médical

Même en mode PRODUCTION_READY avec de vraies données Kaggle :

- Le modèle est **entraîné sur des données publiques non validées cliniquement**
- Les performances rapportées (accuracy ~92% sur Kaggle) **ne reflètent pas** les performances en conditions cliniques réelles
- Ce système reste un **prototype académique non certifié**
- Toute utilisation médicale réelle nécessite validation par un professionnel de santé et certification réglementaire (FDA 510k, CE MDR...)

