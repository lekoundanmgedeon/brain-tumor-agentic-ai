"""
Script de test de la connexion Mistral AI.
Lancez avant de démarrer l'application pour valider votre clé API.

Usage :
    python test_mistral_connection.py

Obtenir une clé API Mistral :
    https://console.mistral.ai/api-keys/
"""

import os
import sys


# ─────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────

def _read_env(key: str) -> str:
    """Lit une valeur depuis les variables d'environnement ou le fichier .env."""
    value = os.getenv(key, "")
    if not value and os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key}="):
                    value = line.split("=", 1)[1].strip()
                    break
    return value


def _patch_env(key: str, value: str):
    """Met à jour ou ajoute une clé=valeur dans .env."""
    if not os.path.exists(".env"):
        return
    with open(".env", "r") as f:
        lines = f.readlines()
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"{key}={value}\n")
    with open(".env", "w") as f:
        f.writelines(new_lines)
    print(f"   ✏️  .env mis à jour : {key}={value}")


def _classify_error(err: str) -> str:
    """Retourne un code d'erreur lisible depuis le message brut."""
    e = err.lower()
    if "401" in err or "unauthorized" in e or "invalid api" in e or "invalid_api" in e:
        return "INVALID_KEY"
    if "403" in err or "forbidden" in e or "allowlist" in e:
        return "FORBIDDEN"
    if "429" in err or "too many" in e or "rate limit" in e or "quota" in e:
        return "RATE_LIMIT"
    if "404" in err or "not found" in e or "does not exist" in e:
        return "NOT_FOUND"
    if "500" in err or "502" in err or "503" in err or "service unavailable" in e:
        return "SERVER_ERROR"
    if "timeout" in e or "timed out" in e:
        return "TIMEOUT"
    if "connect" in e or "network" in e or "connection" in e:
        return "NETWORK"
    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────
# Test principal Mistral
# ─────────────────────────────────────────────────────────────

# Modèles Mistral disponibles, du plus léger au plus puissant.
# Les modèles "open-*" sont gratuits sans restriction de quota.
MISTRAL_MODELS = [
    "mistral-small-latest",     # Rapide, Free Tier, recommandé pour ce projet
    "mistral-small-2503",       # Version datée stable de small
    "open-mistral-nemo",        # Open source, très accessible
    "open-mistral-7b",          # Open source léger
    "mistral-medium-latest",    # Qualité supérieure
    "mistral-large-latest",     # Meilleure qualité, plus lent
]

PROMPT_TEST = (
    "Réponds en une seule phrase : "
    "tu es un assistant médical IA académique dédié à l'analyse d'images IRM cérébrales."
)


def test_mistral() -> bool:
    """
    Teste la connexion à l'API Mistral AI.

    1. Vérifie la présence de MISTRAL_API_KEY
    2. Vérifie l'installation du SDK mistralai
    3. Essaie les modèles dans l'ordre jusqu'à succès
    4. Met à jour .env avec le meilleur modèle disponible

    Returns:
        True si au moins un modèle répond, False sinon
    """
    # ── 1. Clé API ────────────────────────────────────────────
    api_key = _read_env("MISTRAL_API_KEY")
    if not api_key:
        print("❌ MISTRAL_API_KEY non trouvée.")
        print()
        print("   Ajoutez dans votre fichier .env :")
        print("   MISTRAL_API_KEY=votre-cle-mistral")
        print()
        print("   Obtenez une clé gratuite sur : https://console.mistral.ai/api-keys/")
        return False

    print(f"✅ Clé trouvée (longueur : {len(api_key)} chars, "
          f"début : {api_key[:8]}...)")

    # ── 2. SDK ────────────────────────────────────────────────
    try:
        from mistralai.client import Mistral
    except ImportError:
        # Tenter l'ancien import (SDK < 1.0)
        try:
            from mistralai import Mistral  # noqa: F811
        except ImportError:
            print()
            print("❌ SDK mistralai non installé.")
            print("   Installez avec : pip install mistralai")
            return False

    client = Mistral(api_key=api_key)

    # ── 3. Modèle configuré ───────────────────────────────────
    configured_model = _read_env("MISTRAL_MODEL")
    candidates = []
    if configured_model and configured_model not in MISTRAL_MODELS:
        candidates.append(configured_model)   # modèle personnalisé en premier
    elif configured_model:
        candidates.append(configured_model)
    for m in MISTRAL_MODELS:
        if m not in candidates:
            candidates.append(m)

    # ── 4. Tentatives ─────────────────────────────────────────
    print()
    for model in candidates:
        print(f"🔄 Test avec : {model} ...", end=" ", flush=True)
        try:
            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": PROMPT_TEST}],
                temperature=0.1,
                max_tokens=80,
            )
            text = response.choices[0].message.content.strip()
            print("✅")
            print()
            print(f"   Modèle      : {model}")
            print(f"   Réponse     : {text[:140]}")
            usage = response.usage
            if usage:
                print(f"   Tokens      : {usage.prompt_tokens} prompt + "
                      f"{usage.completion_tokens} completion")
            print()

            # Mettre à jour .env si le modèle diffère
            if model != configured_model:
                print(f"⚙️  Modèle différent du configuré ('{configured_model or 'non défini'}') :")
                _patch_env("MISTRAL_MODEL", model)

            print("🟢 Configuration recommandée dans .env :")
            print(f"   APP_MODE=PRODUCTION_READY")
            print(f"   LLM_PROVIDER=mistral")
            print(f"   MISTRAL_API_KEY={api_key[:8]}...")
            print(f"   MISTRAL_MODEL={model}")
            return True

        except Exception as e:
            err_str = str(e)
            code = _classify_error(err_str)

            if code == "RATE_LIMIT":
                print("⚠️  Rate limit — essai du modèle suivant...")
                continue
            elif code == "NOT_FOUND":
                print("⚠️  Modèle non disponible — essai suivant...")
                continue
            elif code == "INVALID_KEY":
                print("❌")
                print()
                print("❌ Clé API Mistral invalide ou expirée.")
                print(f"   Détail : {err_str[:200]}")
                print()
                print("   → Vérifiez sur : https://console.mistral.ai/api-keys/")
                print("   → La clé doit commencer par des caractères alphanumériques")
                return False
            elif code == "FORBIDDEN":
                print("❌")
                print()
                print("❌ Accès refusé (403 Forbidden).")
                print("   → Votre clé est peut-être restreinte à certains modèles.")
                print("   → Vérifiez les permissions sur : https://console.mistral.ai/")
                return False
            elif code == "SERVER_ERROR":
                print("⚠️  Erreur serveur Mistral — essai du modèle suivant...")
                continue
            elif code == "TIMEOUT":
                print("⚠️  Timeout — essai du modèle suivant...")
                continue
            elif code == "NETWORK":
                print("❌")
                print()
                print("❌ Erreur réseau — vérifiez votre connexion Internet.")
                return False
            else:
                print(f"❌ Erreur inattendue : {err_str[:200]}")
                continue

    # ── Tous les modèles ont échoué ───────────────────────────
    print()
    print("❌ Aucun modèle Mistral disponible avec cette clé.")
    print()
    print("Causes possibles :")
    print("  1. Rate limit temporaire")
    print("     → Attendez quelques secondes et relancez")
    print()
    print("  2. Clé API sans accès aux modèles listés")
    print("     → Vérifiez les permissions : https://console.mistral.ai/")
    print()
    print("  3. Quota mensuel épuisé (Free Tier)")
    print("     → Consultez votre usage : https://console.mistral.ai/billing/")
    print("     → Le Free Tier Mistral offre 1$/mois de crédits gratuits")
    print()
    print("💡 L'application fonctionne en mode FALLBACK sans clé LLM.")
    print("   Lancez quand même : streamlit run app.py")
    return False


# ─────────────────────────────────────────────────────────────
# Tests annexes
# ─────────────────────────────────────────────────────────────

def test_sdk_version():
    """Affiche la version installée du SDK Mistral."""
    try:
        import importlib.metadata
        version = importlib.metadata.version("mistralai")
        print(f"✅ SDK mistralai {version} installé")
        # Vérifier que c'est bien une version récente
        major = int(version.split(".")[0])
        if major < 1:
            print("   ⚠️  Version ancienne détectée. Mettez à jour : pip install -U mistralai")
    except Exception:
        print("⚠️  Impossible de déterminer la version du SDK mistralai")


def test_torch():
    """Vérifie la disponibilité de PyTorch pour le mode PRODUCTION_READY."""
    try:
        import torch
        import torchvision
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"✅ PyTorch {torch.__version__} (device : {device})")
        print(f"✅ torchvision {torchvision.__version__}")
        if torch.cuda.is_available():
            print(f"   GPU : {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("⚠️  PyTorch non installé — mode PRODUCTION_READY non disponible")
        print("   Installez avec : pip install torch torchvision")


def test_model_weights():
    """Vérifie la présence des poids du modèle EfficientNet-B0."""
    weights_path = os.path.join("models", "classifier_weights.pth")
    if os.path.exists(weights_path):
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"✅ Poids modèle présents : {weights_path} ({size_mb:.1f} MB)")
    else:
        print(f"⚠️  Poids manquants : {weights_path}")
        print("   Générez-les avec : python models/download_weights.py")


def show_free_tier_info():
    """Affiche les informations sur le Free Tier Mistral."""
    print()
    print("ℹ️  Free Tier Mistral AI :")
    print("   • 1 $/mois de crédits offerts (renouvelés chaque mois)")
    print("   • Modèles accessibles : mistral-small, open-mistral-nemo, open-mistral-7b")
    print("   • Rate limit : ~5 req/sec, 500 000 tokens/mois sur Free Tier")
    print("   • Inscription : https://console.mistral.ai/")
    print("   • Aucune carte bancaire requise pour le Free Tier")


# ─────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 57)
    print("  Brain Tumor AI — Test de Configuration Mistral")
    print("=" * 57)
    print()

    print("── SDK Mistral ───────────────────────────────────────")
    test_sdk_version()
    print()

    print("── PyTorch / Modèle ──────────────────────────────────")
    test_torch()
    test_model_weights()
    print()

    print("── Mistral API ───────────────────────────────────────")
    ok = test_mistral()
    print()

    if not ok:
        show_free_tier_info()
        print()

    print("=" * 57)
    if ok:
        print("✅ Tout est prêt — mode PRODUCTION_READY + Mistral !")
        print()
        print("   Lancez : streamlit run app.py")
    else:
        print("⚠️  Mistral non disponible.")
        print("   L'app fonctionne quand même en mode FALLBACK.")
        print("   Lancez : streamlit run app.py")
    print("=" * 57)
