# ============================================================
# Brain Tumor Agentic AI — Dockerfile
# ============================================================

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Brain Tumor Agentic AI"
LABEL description="Analyse IRM cérébrale agentique — Projet académique"

# Répertoire de travail
WORKDIR /app

# Dépendances système (OpenCV headless + utils)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Création des répertoires nécessaires
RUN mkdir -p uploads outputs/masks outputs/heatmaps outputs/reports

# Variables d'environnement par défaut
ENV APP_MODE=DEMO
ENV LLM_PROVIDER=mock
ENV ENABLE_PUBMED=false
ENV ENABLE_NCI_FALLBACK=true

# Port Streamlit
EXPOSE 8501

# Point de santé
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Lancement de l'application
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
