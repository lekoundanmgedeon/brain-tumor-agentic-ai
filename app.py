"""
Brain Tumor Agentic AI — Application Streamlit
================================================
Interface principale permettant l'analyse IRM cérébrale via
une architecture agentique LangGraph.

⚕️ AVERTISSEMENT : Application académique uniquement.
    Ne constitue pas un diagnostic médical.
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s : %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration de la page ---
st.set_page_config(
    page_title="Brain Tumor AI — Analyse IRM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Imports du projet ---
try:
    from config.settings import settings
    from workflow.graph import run_brain_tumor_analysis
    from services.visualization import create_analysis_figure, create_metrics_chart
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    st.error(f"❌ Erreur d'import : {e}. Vérifiez l'installation des dépendances.")
    st.stop()


# ============================================================
# STYLES CSS PERSONNALISÉS
# ============================================================
st.markdown("""
<style>
    .medical-warning {
        background: linear-gradient(135deg, #1a0a0a, #2d0d0d);
        border: 2px solid #ff4b4b;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        color: #ffcdd2;
    }
    .mode-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: bold;
        margin: 4px 0;
    }
    .badge-demo { background: #1a3a1a; color: #4caf50; border: 1px solid #4caf50; }
    .badge-gemini { background: #1a2a3a; color: #2196f3; border: 1px solid #2196f3; }
    .badge-mistral { background: #2a1a3a; color: #9c27b0; border: 1px solid #9c27b0; }
    .badge-fallback { background: #2a2a1a; color: #ff9800; border: 1px solid #ff9800; }
    .metric-card {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #333;
        text-align: center;
    }
    .finding-item {
        padding: 4px 0;
        color: #b0bec5;
        font-size: 14px;
    }
    .report-container {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 20px;
        font-family: 'Georgia', serif;
        line-height: 1.7;
    }
    .source-item {
        padding: 4px 0;
        font-size: 13px;
        color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    """Affiche la barre latérale avec la configuration."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
        st.title("🧠 Brain Tumor AI")
        st.caption("Analyse IRM Agentique — v1.0")

        st.divider()

        # --- Mode application ---
        st.subheader("⚙️ Configuration")
        provider = settings.get_llm_provider()
        mode_map = {
            "gemini": ("badge-gemini", "🔵 GEMINI"),
            "mistral": ("badge-mistral", "🟣 MISTRAL"),
            "mock": ("badge-fallback", "🟠 FALLBACK"),
        }
        css_class, label = mode_map.get(provider, ("badge-demo", "🟢 DEMO"))
        st.markdown(f'<span class="mode-badge {css_class}">{label}</span>', unsafe_allow_html=True)

        app_mode_label = f'<span class="mode-badge badge-demo">Mode : {settings.APP_MODE}</span>'
        st.markdown(app_mode_label, unsafe_allow_html=True)

        st.divider()

        # --- Contexte patient (optionnel) ---
        st.subheader("👤 Contexte Patient (Optionnel)")
        patient_age = st.number_input("Âge", min_value=0, max_value=120, value=0, step=1)
        patient_sex = st.selectbox("Sexe", ["Non précisé", "Masculin", "Féminin"])
        patient_symptoms = st.text_area(
            "Symptômes rapportés",
            placeholder="Ex: céphalées, troubles visuels...",
            height=80,
        )

        patient_context = {}
        if patient_age > 0:
            patient_context["age"] = patient_age
        if patient_sex != "Non précisé":
            patient_context["sexe"] = patient_sex
        if patient_symptoms:
            patient_context["symptomes"] = patient_symptoms

        st.divider()

        # --- Paramètres avancés ---
        st.subheader("🔧 Paramètres")
        show_technical = st.checkbox("Afficher les détails techniques", value=True)
        show_sources = st.checkbox("Afficher les sources RAG", value=True)

        st.divider()
        st.caption("⚕️ Ce système est un prototype académique. Aucune certification médicale.")
        st.caption("© 2024 — Brain Tumor Agentic AI")

    return patient_context, show_technical, show_sources


# ============================================================
# AVERTISSEMENT MÉDICAL
# ============================================================
def render_medical_warning():
    """Affiche l'avertissement médical en haut de page."""
    st.markdown("""
    <div class="medical-warning">
        <strong>⚕️ AVERTISSEMENT MÉDICAL IMPORTANT</strong><br>
        Cette application est un prototype académique à des fins d'exploration de l'IA médicale.
        <strong>Elle ne constitue en aucun cas un diagnostic médical définitif</strong> et ne remplace
        pas l'expertise d'un radiologue, d'un neurologue ou de tout professionnel de santé qualifié.
        Toute décision médicale doit être prise par un professionnel de santé après examen clinique complet.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SECTION RÉSULTATS
# ============================================================
def render_results(result: dict, show_technical: bool, show_sources: bool):
    """Affiche les résultats de l'analyse."""

    tumor_detected = result.get("tumor_detected", False)
    tumor_type = result.get("suspected_tumor_type", "unknown") or "unknown"
    confidence = result.get("confidence", 0.0) or 0.0
    confidence_level = result.get("confidence_level", "low") or "low"
    location = result.get("tumor_location") or "Non déterminée"
    area = result.get("tumor_area_mm2")
    mask_path = result.get("segmentation_mask_path")
    heatmap_path = result.get("heatmap_path")
    image_path = result.get("image_path", "")
    warnings = result.get("validation_warnings", [])
    findings = result.get("technical_findings", [])
    final_report = result.get("final_report", "")
    sources = result.get("sources", [])
    llm_provider = result.get("llm_provider_used", "FALLBACK")
    quality_score = result.get("image_quality_score", 0.0) or 0.0

    # --- Statut principal ---
    st.divider()
    if not result.get("image_valid", True):
        st.error("❌ Image invalide — Analyse impossible")
        for err in result.get("image_errors", []):
            st.warning(f"• {err}")
    elif tumor_detected:
        st.error(f"⚠️ Anomalie suspecte détectée — {tumor_type.replace('_', ' ').title()}")
    else:
        st.success("✅ Aucune anomalie détectée par l'IA")

    # --- Avertissements de validation ---
    if warnings:
        with st.expander(f"⚠️ {len(warnings)} avertissement(s) de validation", expanded=False):
            for w in warnings:
                st.warning(w)

    # --- Métriques clés ---
    st.subheader("📊 Métriques d'Analyse")
    col1, col2, col3, col4 = st.columns(4)

    confidence_color = "🟢" if confidence >= 0.75 else "🟡" if confidence >= 0.4 else "🔴"
    with col1:
        st.metric("Confiance IA", f"{confidence:.1%}", help="Score de confiance du modèle de détection")
    with col2:
        st.metric("Niveau Confiance", f"{confidence_color} {confidence_level.upper()}")
    with col3:
        st.metric("Qualité Image", f"{quality_score:.1%}")
    with col4:
        if area:
            st.metric("Surface Estimée", f"{area:.1f} mm²")
        else:
            st.metric("Surface Estimée", "N/A")

    # --- Graphique métriques ---
    try:
        metrics_chart = create_metrics_chart(confidence, quality_score, area)
        st.image(metrics_chart, use_container_width=False, width=500)
    except Exception as e:
        logger.warning(f"Erreur graphique métriques : {e}")

    # --- Visualisations ---
    st.subheader("🔬 Visualisations")
    try:
        if os.path.exists(image_path):
            fig_bytes = create_analysis_figure(
                original_path=image_path,
                mask_path=mask_path,
                heatmap_path=heatmap_path,
                tumor_detected=tumor_detected,
                tumor_type=tumor_type,
                confidence=confidence,
            )
            st.image(fig_bytes, use_container_width=True)
    except Exception as e:
        logger.warning(f"Erreur figure principale : {e}")
        st.warning(f"Impossible de générer la visualisation : {e}")

    # Images individuelles
    img_cols = []
    if mask_path and os.path.exists(mask_path):
        img_cols.append(("Masque de Segmentation", mask_path))
    if heatmap_path and os.path.exists(heatmap_path):
        img_cols.append(("Heatmap d'Attention", heatmap_path))

    if img_cols:
        cols = st.columns(len(img_cols))
        for col, (title, path) in zip(cols, img_cols):
            with col:
                st.caption(title)
                st.image(path, use_container_width=True)

    # --- Détails techniques ---
    if show_technical:
        with st.expander("🔧 Détails Techniques", expanded=False):
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Observations du modèle :**")
                for f in findings:
                    st.markdown(f'<div class="finding-item">• {f}</div>', unsafe_allow_html=True)
            with col_r:
                st.markdown("**Localisation approx. :**")
                st.info(location)
                if area:
                    st.markdown(f"**Surface tumorale :** {area:.1f} mm²")
                st.markdown(f"**Mode LLM :** `{llm_provider}`")
                st.markdown(f"**Mode App :** `{settings.APP_MODE}`")

    # --- Rapport Final ---
    st.subheader("📋 Rapport d'Analyse IA")
    provider_label = {
        "GEMINI": "🔵 Généré par Gemini",
        "MISTRAL": "🟣 Généré par Mistral",
        "FALLBACK": "🟠 Mode Fallback (sans LLM externe)",
    }.get(llm_provider, f"Mode : {llm_provider}")

    st.caption(provider_label)

    if final_report:
        st.markdown(
            f'<div class="report-container">{final_report.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Aucun rapport généré.")

    # --- Sources RAG ---
    if show_sources and sources:
        st.subheader("📚 Sources Utilisées")
        for s in sources:
            if s.startswith("http"):
                st.markdown(f'<div class="source-item">🔗 <a href="{s}" target="_blank">{s}</a></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="source-item">📄 {s}</div>', unsafe_allow_html=True)

    # --- Téléchargement du rapport ---
    st.divider()
    st.subheader("💾 Télécharger le Rapport")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        if final_report:
            st.download_button(
                label="📄 Télécharger en TXT",
                data=final_report.encode("utf-8"),
                file_name=f"rapport_IA_IRM_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

    with col_dl2:
        # Export JSON des résultats techniques
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "tumor_detected": tumor_detected,
            "suspected_tumor_type": tumor_type,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "tumor_location": location,
            "tumor_area_mm2": area,
            "technical_findings": findings,
            "validation_warnings": warnings,
            "sources": sources,
            "llm_provider": llm_provider,
            "disclaimer": "Ce rapport est généré par un système IA académique. Ne constitue pas un diagnostic médical.",
        }
        st.download_button(
            label="📊 Télécharger en JSON",
            data=json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"resultats_IA_IRM_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


# ============================================================
# APPLICATION PRINCIPALE
# ============================================================
def main():
    """Point d'entrée principal de l'application Streamlit."""

    # --- Avertissement médical ---
    render_medical_warning()

    # --- Titre ---
    st.title("🧠 Brain Tumor Agentic AI")
    st.subheader("Analyse IRM Cérébrale par Architecture Agentique LangGraph")

    # --- Sidebar ---
    patient_context, show_technical, show_sources = render_sidebar()

    # --- Upload de l'image ---
    st.divider()
    st.subheader("📤 Charger une Image IRM")

    uploaded_file = st.file_uploader(
        "Sélectionnez une image IRM cérébrale (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        help="L'image doit être une coupe axiale, coronale ou sagittale d'IRM cérébrale.",
    )

    if uploaded_file is not None:
        # --- Affichage de l'image originale ---
        col_preview, col_info = st.columns([2, 1])

        with col_preview:
            st.subheader("🖼️ Image Sélectionnée")
            img = Image.open(uploaded_file)
            st.image(img, caption=f"Image : {uploaded_file.name}", use_container_width=True)

        with col_info:
            st.subheader("📋 Informations Fichier")
            st.markdown(f"**Nom :** `{uploaded_file.name}`")
            st.markdown(f"**Type :** `{uploaded_file.type}`")
            st.markdown(f"**Taille :** `{uploaded_file.size / 1024:.1f} Ko`")
            w, h = img.size
            st.markdown(f"**Dimensions :** `{w} × {h} px`")

            if patient_context:
                st.markdown("**Contexte patient :**")
                for k, v in patient_context.items():
                    st.markdown(f"• {k} : {v}")

        # --- Bouton d'analyse ---
        st.divider()

        if st.button(
            "🚀 Lancer l'Analyse Agentique",
            type="primary",
            use_container_width=True,
            help="Déclenche le pipeline complet : Validation → Radiologue → RAG → Rapport",
        ):
            # Sauvegarder l'image dans le dossier uploads/
            settings.ensure_directories()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = f"irm_{timestamp}_{uploaded_file.name.replace(' ', '_')}"
            image_save_path = os.path.join(settings.UPLOAD_DIR, safe_name)

            with open(image_save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # --- Pipeline d'analyse ---
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            steps = [
                ("🔍 Validation de l'image...", 0.15),
                ("🧠 Analyse radiologique IA...", 0.40),
                ("✅ Validation des résultats...", 0.60),
                ("📚 Recherche documentaire (RAG)...", 0.80),
                ("📝 Génération du rapport...", 0.95),
            ]

            progress_bar = st.progress(0.0)
            step_text = st.empty()

            # Simuler l'affichage des étapes
            import time

            with st.spinner("⏳ Pipeline agentique en cours..."):
                for step_msg, prog in steps:
                    step_text.info(step_msg)
                    progress_bar.progress(prog)
                    time.sleep(0.3)  # Petit délai pour l'UX

                try:
                    # Lancer l'analyse complète
                    result = run_brain_tumor_analysis(
                        image_path=image_save_path,
                        patient_context=patient_context if patient_context else None,
                    )
                    progress_bar.progress(1.0)
                    step_text.success("✅ Analyse terminée !")

                except Exception as e:
                    progress_bar.empty()
                    step_text.empty()
                    st.error(f"❌ Erreur lors de l'analyse : {str(e)}")
                    logger.error(f"Erreur pipeline : {e}", exc_info=True)
                    result = None

            # --- Affichage des résultats ---
            if result:
                render_results(result, show_technical, show_sources)

    else:
        # --- Page d'accueil ---
        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 🔬 Détection
            Identification automatique des zones suspectes sur l'image IRM
            grâce à des algorithmes de vision par ordinateur.
            """)

        with col2:
            st.markdown("""
            ### 🎯 Segmentation
            Délimitation précise de la région d'intérêt avec génération
            d'un masque et d'une heatmap d'attention.
            """)

        with col3:
            st.markdown("""
            ### 📚 RAG Médical
            Enrichissement automatique avec des informations médicales
            provenant de PubMed et du National Cancer Institute.
            """)

        st.divider()
        st.info(
            "👆 **Commencez par charger une image IRM cérébrale** (PNG, JPG ou JPEG) "
            "pour lancer l'analyse agentique complète.\n\n"
            "💡 Vous pouvez utiliser l'image de démonstration dans `demo_assets/sample_mri.png`."
        )

        # --- Architecture du workflow ---
        with st.expander("🗺️ Architecture du Workflow Agentique", expanded=False):
            st.markdown("""
            ```
            📤 Image IRM (Streamlit)
                    ↓
            🔍 Agent Validation Image
              → Vérification format, taille, qualité
                    ↓ (si valide)
            🧠 Agent Radiologue (Vision)
              → Prétraitement → Inférence → Segmentation
                    ↓
            ✅ Agent Validation Résultats
              → Vérification confiance, masque, cohérence
                    ↓ (si fiable)
            📚 Agent RAG Médical
              → PubMed + NCI + Base locale
                    ↓
            📝 Agent Rapport (Gemini/Mistral/Fallback)
              → Rapport structuré prudent
                    ↓
            📋 Rapport Final (Streamlit)
            ```
            """)


if __name__ == "__main__":
    main()
