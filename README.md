# 🧠 Brain Tumor Agentic AI

> **Détection, segmentation, diagnostic assisté et recommandations autour des tumeurs cérébrales à partir d'images IRM.**
>
> Projet académique — Architecture agentique basée sur LangGraph, Streamlit, RAG médical et LLM (Gemini/Mistral).

---

## ⚕️ Avertissement Médical Important

**Cette application est un prototype académique à des fins d'exploration de l'IA médicale.**

Elle ne constitue en aucun cas un diagnostic médical définitif et **ne remplace pas** l'expertise d'un radiologue, d'un neurologue ou de tout professionnel de santé qualifié. Toute décision médicale doit être prise par un professionnel de santé après examen clinique complet.

---

## 1. Présentation du Projet

Brain Tumor Agentic AI est une application Streamlit qui analyse des images IRM cérébrales via une architecture multi-agents orchestrée par LangGraph.

L'application :
- Valide l'image IRM soumise
- Détecte et segmente les zones suspectes
- Interroge PubMed et une base médicale locale
- Génère un rapport structuré et prudent via Gemini, Mistral ou un fallback local

---

## 2. Architecture Agentique

```
📤 Image IRM (Streamlit)
        ↓
🔍 Agent Validation Image
  → Format, taille, qualité
        ↓ (si valide)
🧠 Agent Radiologue (Vision)
  → Prétraitement → Inférence → Segmentation
        ↓
✅ Agent Validation Résultats
  → Confiance, masque, cohérence
        ↓ (si fiable)
📚 Agent RAG Médical
  → PubMed + NCI + Base locale
        ↓
📝 Agent Rapport (Gemini/Mistral/Fallback)
  → Rapport structuré prudent avec sources
        ↓
📋 Rapport Final (Streamlit)
```

**Transitions conditionnelles :**
- Image invalide → Rapport d'erreur direct
- Résultat incertain → Rapport prudent sans RAG
- Résultat fiable → Pipeline complet

---

## 3. Description des Agents

| Agent | Mission |
|-------|---------|
| **Orchestrateur** | Initialise l'état global et contrôle les transitions |
| **Validation Image** | Vérifie format, taille, qualité de l'image IRM |
| **Radiologue (Vision)** | Détecte la tumeur, génère masque et heatmap |
| **Validation Résultats** | Vérifie confiance, masque, cohérence du résultat |
| **RAG Médical** | Interroge PubMed et la base de connaissances locale |
| **Rapport** | Génère le rapport via Gemini, Mistral ou fallback |

---

## 4. Installation Locale

### Prérequis
- Python 3.10 ou 3.11
- pip

### Linux / macOS

```bash
git clone https://github.com/votre-repo/brain-tumor-agentic-ai.git
cd brain-tumor-agentic-ai

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Windows

```cmd
git clone https://github.com/votre-repo/brain-tumor-agentic-ai.git
cd brain-tumor-agentic-ai

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## 5. Configuration des Clés API

```bash
cp .env.example .env
```

Éditez le fichier `.env` :

```env
APP_MODE=DEMO
LLM_PROVIDER=mock

# Décommentez et remplissez selon votre provider :
# GEMINI_API_KEY=votre-cle-gemini
# MISTRAL_API_KEY=votre-cle-mistral

PUBMED_EMAIL=votre-email@example.com
ENABLE_PUBMED=false
ENABLE_NCI_FALLBACK=true
```

**Obtenir les clés :**
- Google Gemini : https://aistudio.google.com/
- Mistral AI : https://console.mistral.ai/

---

## 6. Lancement de l'Application

```bash
# Activer l'environnement virtuel (si pas déjà fait)
source venv/bin/activate      # Linux/macOS
# ou
venv\Scripts\activate         # Windows

# Lancer Streamlit
streamlit run app.py
```

L'application s'ouvre automatiquement sur : **http://localhost:8501**

---

## 7. Mode DEMO

Le mode DEMO est le mode par défaut. Il ne nécessite **aucune clé API** et **aucun modèle médical lourd**.

Fonctionnement :
- La détection est simulée à partir des statistiques de l'image (variance, intensité)
- La segmentation génère une ellipse réaliste sur la zone suspecte
- Le rapport est généré par un template structuré (mode fallback)
- La base de connaissances locale est utilisée pour le RAG

Pour tester en mode DEMO :
1. Lancez l'application : `streamlit run app.py`
2. Uploadez n'importe quelle image PNG/JPG
3. Cliquez "Lancer l'Analyse"

Une image de démonstration est disponible dans `demo_assets/sample_mri.png`.

---

## 8. Mode Gemini / Mistral

Pour utiliser un LLM externe dans `.env` :

**Gemini :**
```env
GEMINI_API_KEY=votre-cle-ici
LLM_PROVIDER=gemini
```

**Mistral :**
```env
MISTRAL_API_KEY=votre-cle-ici
LLM_PROVIDER=mistral
```

Le rapport sera alors généré par le LLM configuré, offrant une analyse plus nuancée et contextuelle.

**Priorité de sélection automatique :** Gemini > Mistral > Fallback

---

## 9. Déploiement Docker

```bash
# Construire et lancer avec docker compose
docker compose up --build

# En arrière-plan
docker compose up --build -d

# Voir les logs
docker compose logs -f

# Arrêter
docker compose down
```

L'application sera accessible sur : **http://localhost:8501**

Pour personnaliser la configuration Docker :

```bash
# Modifier les variables d'environnement dans docker-compose.yml
# ou créer un fichier .env à la racine du projet
```

---

## 10. Limites Médicales

Cette application présente les limitations suivantes qui doivent être comprises avant tout usage :

1. **Pas de certification médicale** : Le système n'est certifié par aucune autorité de santé (FDA, CE, ANSM...).

2. **Analyse 2D uniquement** : Une IRM clinique utilise des coupes 3D multiséquences avec injection de contraste.

3. **Faux positifs et faux négatifs possibles** : Tout modèle d'IA produit des erreurs ; un résultat négatif ne garantit pas l'absence de tumeur.

4. **Pas de biopsie virtuelle** : La classification histologique nécessite une biopsie et une analyse anatomopathologique.

5. **Mode DEMO simulé** : En mode DEMO, les résultats sont basés sur une heuristique statistique et non sur un vrai modèle entraîné.

6. **Données d'entraînement inconnues** : En mode DEMO, aucun entraînement sur des données médicales réelles n'a été effectué.

**Ce système ne doit jamais être utilisé pour prendre une décision médicale.**

---

## 11. Brancher un Vrai Modèle (MONAI / nnU-Net / MedSAM)

Le projet est conçu pour faciliter l'intégration de modèles médicaux réels.

### MONAI

1. Installez MONAI : `pip install monai torch torchvision`
2. Modifiez `services/inference.py`, fonction `_run_production_inference` :

```python
from monai.networks.nets import UNet
import torch

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=4,  # background + 3 types de tumeurs
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
).eval()
model.load_state_dict(torch.load("path/to/weights.pth", map_location="cpu"))

with torch.no_grad():
    input_tensor = torch.from_numpy(preprocessed_image).unsqueeze(0)
    output = torch.softmax(model(input_tensor), dim=1)
    pred_class = output.argmax(dim=1).item()
```

### nnU-Net v2

1. Installez : `pip install nnunetv2`
2. Dans `services/inference.py` :

```python
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True)
predictor.initialize_from_trained_model_folder(
    "path/to/nnunet/model",
    use_folds=(0,),
    checkpoint_name="checkpoint_best.pth",
)
```

### MedSAM

1. Installez : `pip install git+https://github.com/facebookresearch/segment-anything.git`
2. Dans `services/segmentation.py` :

```python
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_b"](checkpoint="medsam_vit_b.pth")
predictor = SamPredictor(sam)
predictor.set_image(image_array)
masks, _, _ = predictor.predict(point_coords=..., point_labels=...)
```

Définissez `APP_MODE=PRODUCTION_READY` dans `.env` pour activer le mode production.

---

## 12. Structure du Projet

```
brain-tumor-agentic-ai/
│
├── app.py                          # Application Streamlit principale
├── requirements.txt                # Dépendances Python
├── README.md                       # Documentation
├── .env.example                    # Template de configuration
├── Dockerfile                      # Image Docker
├── docker-compose.yml              # Orchestration Docker
│
├── agents/                         # Agents LangGraph
│   ├── orchestrator_agent.py       # Agent orchestrateur + routage
│   ├── validation_image_agent.py   # Validation de l'image IRM
│   ├── radiology_agent.py          # Analyse radiologique (vision)
│   ├── validation_results_agent.py # Validation des résultats
│   ├── medical_rag_agent.py        # RAG médical (PubMed + local)
│   └── report_agent.py             # Génération du rapport LLM
│
├── workflow/                       # Définition du workflow LangGraph
│   ├── state.py                    # TypedDict BrainTumorState
│   └── graph.py                    # Graphe LangGraph compilé
│
├── services/                       # Services techniques
│   ├── preprocessing.py            # Prétraitement d'images
│   ├── inference.py                # Détection (DEMO + PRODUCTION)
│   ├── segmentation.py             # Segmentation (masque + heatmap)
│   ├── visualization.py            # Figures Matplotlib
│   ├── rag_pubmed.py               # Client PubMed
│   ├── rag_nci.py                  # NCI + fichiers locaux
│   └── llm_client.py               # Client LLM unifié
│
├── config/
│   └── settings.py                 # Configuration centralisée
│
├── knowledge_base/                 # Base de connaissances médicales
│   ├── glioma.md                   # Gliomes
│   ├── meningioma.md               # Méningiomes
│   └── pituitary_tumor.md          # Adénomes hypophysaires
│
├── uploads/                        # Images IRM uploadées
├── outputs/                        # Sorties générées
│   ├── masks/                      # Masques de segmentation
│   ├── heatmaps/                   # Heatmaps d'attention
│   └── reports/                    # Rapports exportés
│
├── tests/                          # Tests unitaires
│   ├── test_validation.py          # Tests validation image
│   ├── test_workflow.py            # Tests workflow complet
│   └── test_rag.py                 # Tests RAG
│
└── demo_assets/
    └── sample_mri.png              # Image IRM de démonstration
```

---

## Lancer les Tests

```bash
# Tous les tests
pytest tests/ -v

# Un module spécifique
pytest tests/test_validation.py -v
pytest tests/test_workflow.py -v
pytest tests/test_rag.py -v
```

---

## Technologies Utilisées

| Composant | Technologie |
|-----------|-------------|
| Interface | Streamlit |
| Workflow | LangGraph |
| Vision | OpenCV, Pillow, NumPy, Matplotlib |
| RAG | PubMed NCBI API, fichiers Markdown locaux |
| LLM | Google Gemini / Mistral AI / Fallback local |
| Config | python-dotenv |
| Conteneur | Docker + docker-compose |
| Tests | pytest |
| Production (optionnel) | MONAI, nnU-Net, MedSAM, PyTorch |

---

*Projet académique — Ne pas utiliser à des fins médicales réelles.*
