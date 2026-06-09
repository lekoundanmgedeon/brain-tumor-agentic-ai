"""
Script de test rapide de la connexion Gemini.
Lancez avant de démarrer l'application pour valider votre clé API.

Usage :
    python test_gemini_connection.py
"""
import os
import sys

def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        if os.path.exists(".env"):
            with open(".env") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break

    if not api_key:
        print("❌ GEMINI_API_KEY non trouvée dans .env")
        print("   Ajoutez : GEMINI_API_KEY=votre-cle dans le fichier .env")
        sys.exit(1)

    print(f"✅ Clé trouvée (longueur : {len(api_key)} chars)")

    # Lire le modèle configuré, ou tenter une liste de fallbacks
    configured_model = os.getenv("GEMINI_MODEL", "")
    if not configured_model:
        if os.path.exists(".env"):
            with open(".env") as f:
                for line in f:
                    if line.startswith("GEMINI_MODEL="):
                        configured_model = line.split("=", 1)[1].strip()
                        break

    # Ordre de tentative : modèle configuré d'abord, puis fallbacks
    candidates = []
    if configured_model:
        candidates.append(configured_model)
    for m in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
        if m not in candidates:
            candidates.append(m)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("⚠️  SDK google-genai non installé.")
        print("   Installez avec : pip install google-genai")
        return False

    client = genai.Client(api_key=api_key)

    for model in candidates:
        print(f"🔄 Test avec le modèle : {model} ...")
        try:
            response = client.models.generate_content(
                model=model,
                contents="Réponds en une phrase : tu es un assistant médical IA académique.",
                config=types.GenerateContentConfig(max_output_tokens=60, temperature=0.1),
            )
            print(f"✅ Connexion Gemini réussie avec : {model}")
            print(f"   Réponse : {response.text.strip()[:120]}")
            print()
            if model != configured_model:
                print(f"⚙️  Mettez à jour votre .env :")
                print(f"   GEMINI_MODEL={model}")
                _patch_env("GEMINI_MODEL", model)
            print("🟢 Configuration recommandée pour .env :")
            print(f"   APP_MODE=PRODUCTION_READY")
            print(f"   LLM_PROVIDER=gemini")
            print(f"   GEMINI_MODEL={model}")
            return True

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                print(f"   ⚠️  Quota épuisé sur {model} — essai du modèle suivant...")
            elif "invalid" in err.lower() or "api_key" in err.lower():
                print(f"❌ Clé API invalide : {err[:120]}")
                print("   → Vérifiez GEMINI_API_KEY sur https://aistudio.google.com/")
                return False
            elif "not found" in err.lower() or "404" in err:
                print(f"   ⚠️  Modèle {model} non disponible — essai suivant...")
            else:
                print(f"   ❌ Erreur inattendue sur {model} : {err[:200]}")

    # Tous les modèles ont échoué
    print()
    print("❌ Aucun modèle Gemini disponible avec votre clé.")
    print()
    print("Causes possibles :")
    print("  1. Quota Free Tier journalier épuisé sur tous les modèles")
    print("     → Attendez le reset (minuit heure du Pacifique / ~8h Paris)")
    print("     → Ou activez la facturation : https://console.cloud.google.com/billing")
    print()
    print("  2. Clé API incorrecte ou révoquée")
    print("     → Vérifiez sur : https://aistudio.google.com/apikey")
    print()
    print("  3. Projet Google Cloud sans quota actif")
    print("     → https://console.cloud.google.com/iam-admin/quotas")
    print()
    print("💡 En attendant, l'application fonctionne en mode FALLBACK")
    print("   (rapport structuré sans LLM) — lancez quand même : streamlit run app.py")
    return False


def _patch_env(key: str, value: str):
    """Met à jour ou ajoute une clé dans .env si le fichier existe."""
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


def test_torch():
    try:
        import torch
        import torchvision
        print(f"✅ PyTorch {torch.__version__} (device: {'CUDA' if torch.cuda.is_available() else 'CPU'})")
        print(f"✅ torchvision {torchvision.__version__}")
    except ImportError:
        print("⚠️  PyTorch non installé — mode PRODUCTION_READY non disponible")
        print("   Installez avec : pip install torch torchvision")


def test_model_weights():
    weights_path = os.path.join("models", "classifier_weights.pth")
    if os.path.exists(weights_path):
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"✅ Poids modèle présents : {weights_path} ({size_mb:.1f} MB)")
    else:
        print(f"⚠️  Poids manquants : {weights_path}")
        print("   Générez-les avec : python models/download_weights.py")


if __name__ == "__main__":
    print("=" * 55)
    print("  Brain Tumor AI — Test de Configuration Production")
    print("=" * 55)
    print()

    print("── PyTorch / Modèle ──────────────────────────────────")
    test_torch()
    test_model_weights()
    print()

    print("── Gemini API ────────────────────────────────────────")
    ok = test_gemini()
    print()
    print("=" * 55)
    if ok:
        print("✅ Tout est prêt pour le mode PRODUCTION_READY + Gemini !")
    else:
        print("⚠️  Corrigez les erreurs ci-dessus avant de continuer.")
    print("=" * 55)
