# main.py
import streamlit as st
import os
import json
import traceback
from PIL import Image
import torch
import pytesseract
import numpy as np
from pdf2image import convert_from_path
from transformers import LayoutLMv3Processor, LayoutLMv3Model
from sklearn.cluster import KMeans
from ocr.extract_text import extract_text_from_image, extract_text_from_pdf
from models.nlp_extraction import extract_skills_with_spacy
# ---------------------------
# CONFIGURATION GLOBALE
# ---------------------------
st.set_page_config(page_title="🧠 Smart CV Parser", layout="wide")
st.title("🧠 Smart CV Parser")

POPLER_PATH = r"C:\Release-25.07.0-0\poppler-25.07.0\Library\bin"
SKILLS_PATH = "data/employment_skills.txt"

os.makedirs("data/Resume", exist_ok=True)
os.makedirs("outputs/json", exist_ok=True)


# ---------------------------
# DÉTECTION DYNAMIQUE DES SECTIONS LayoutLMv3
# ---------------------------
from models.layoutlm_detect_sections import detect_sections

st.sidebar.success("✅ Module dynamique de détection de sections chargé")

# ---------------------------
# FONCTIONS UTILITAIRES
# ---------------------------
def load_skills(txt_path):
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]



# ---------------------------
# INTERFACE STREAMLIT
# ---------------------------
uploaded_file = st.file_uploader("📄 Téléversez un CV (PDF ou image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    try:
        save_path = os.path.join("data/Resume", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Fichier enregistré : {save_path}")

        ext = uploaded_file.name.split(".")[-1].lower()
        if ext in ["png", "jpg", "jpeg"]:
            st.image(Image.open(save_path), use_container_width=True)
            text = extract_text_from_image(save_path)
        else:
            st.info("PDF uploadé — ")
            text = extract_text_from_pdf(save_path, poppler_path=POPLER_PATH)

        st.subheader("📝 Texte extrait")
        st.text_area("Résultat OCR", text[:5000], height=300)

        skills_list = load_skills(SKILLS_PATH)

        # ---------------------------
        # ---------------------------
        # 1️⃣ Détection dynamique des sections LayoutLMv3
        # ---------------------------
        st.subheader("📚 Sections détectées dynamiquement")

        try:
            with st.spinner("🔍 Détection automatique du nombre optimal de sections..."):
               sections = detect_sections(text)  # pas besoin de n_clusters ici
            st.success("✅ Sections détectées avec succès")

            for s in sections:
               with st.expander(f"📖 {s['section_name']}"):
                   st.write(s['content'][:1000])  # affiche un extrait
        except Exception as e:
            st.error("Erreur lors de la détection des sections")
            st.text(traceback.format_exc())


        # ---------------------------
        # 2️⃣ Détection des compétences via spaCy
        # ---------------------------
        st.subheader("🧩 Compétences détectées ")
        try:
            skills_ner = extract_skills_with_spacy(text)
            st.write(skills_ner if skills_ner else "Aucune compétence détectée.")
        except Exception:
            st.error("Erreur spaCy NER")
            st.text(traceback.format_exc())

        # ---------------------------
        # 3️⃣ Détection des compétences par correspondance simple
        # ---------------------------
        #st.subheader("🧠 Compétences trouvées par matching")
        #skills_match = extract_skills_by_list(text, skills_list)
        #st.write(skills_match if skills_match else "Aucune correspondance trouvée.")

        # ---------------------------
        # 4️⃣ Sauvegarde des résultats
        # ---------------------------
        out = {
            "filename": uploaded_file.name,
            "text_snippet": text[:2000],
            "predicted_section": section_pred if "section_pred" in locals() else None,
            "skills_ner": skills_ner if "skills_ner" in locals() else [],
            #"skills_match": skills_match,
        }
        out_path = os.path.join("outputs/json", uploaded_file.name + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        st.success(f"📁 Résultats sauvegardés dans {out_path}")

    except Exception:
        st.error("Erreur principale")
        st.text(traceback.format_exc())
