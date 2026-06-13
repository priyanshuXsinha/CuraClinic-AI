"""
streamlit_app.py — Streamlit web UI for the AI Medical Expert (Phi-2 fine-tuned).

Run locally:
    streamlit run streamlit_app.py

Deploy on Streamlit Cloud:
    Push this file + requirements_streamlit.txt to GitHub.
    Go to share.streamlit.io → New app → select this repo.
    Add HF_TOKEN in Secrets (Settings → Secrets).

Environment variable / Streamlit Secret:
    HF_TOKEN  — required if model repo is private
"""

import os
import torch
import streamlit as st
from threading import Thread
from transformers import TextIteratorStreamer
from unsloth import FastLanguageModel

# ──────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Medical Expert",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_REPO   = "priyanshuxsinha/phi2-medical-expert"
LOAD_IN_4BIT = True
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", None))

INFERENCE_PROMPT = (
    "### Instruction:\nAnswer as a medical expert.\n\n"
    "### Question:\n{question}\n\n"
    "### Response:\n<think>\n"
)

SAMPLE_CASES = [
    "A 61-year-old woman has involuntary urine loss when coughing but no leakage at night. "
    "What would cystometry reveal about residual volume and detrusor contractions?",

    "A 59-year-old man has fever, night sweats, and a 12 mm aortic valve vegetation. "
    "Blood cultures show gram-positive, catalase-negative, gamma-hemolytic cocci in chains "
    "that do not grow in 6.5% NaCl. What is the most likely predisposing factor?",

    "Explain the pathophysiology of Type 2 Diabetes Mellitus and its recommended first-line treatment.",
]

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #0d5c63;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 24px;
    }
    .response-box {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        border-radius: 12px;
        padding: 20px;
        min-height: 300px;
        font-size: 15px;
        line-height: 1.7;
        color: #1e293b;
        white-space: pre-wrap;
    }
    .disclaimer {
        background: #fef9c3;
        border-left: 4px solid #eab308;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #713f12;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        border-top: 1px solid #e2e8f0;
        padding-top: 16px;
        margin-top: 32px;
    }
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load model (cached — only loads once per session)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading Phi-2 medical model — please wait …")
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_REPO,
        load_in_4bit=LOAD_IN_4BIT,
        token=HF_TOKEN,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

# ──────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────
def run_inference_stream(model, tokenizer, question: str):
    prompt = INFERENCE_PROMPT.format(question=question)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=1024, use_cache=True)

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for token in streamer:
        yield token

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=64)
    st.markdown("## ⚙️ Settings")
    st.slider("Max response tokens", 256, 1500, 1024, 64, key="max_tokens")
    st.divider()

    st.markdown("### 📋 Sample Cases")
    for i, case in enumerate(SAMPLE_CASES, 1):
        if st.button(f"Case {i}", key=f"sample_{i}", use_container_width=True):
            st.session_state["prefill"] = case

    st.divider()
    st.markdown("""
**Model:** `phi2-medical-expert`  
**Base:** Microsoft Phi-2  
**Fine-tuning:** Unsloth + LoRA  
**Dataset:** Medical-o1 Reasoning SFT  
    """)

# ──────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────
st.markdown("<h1 class='main-title'>🏨 Clinical Reasoning Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced medical chain-of-thought reasoning powered by Phi-2</p>", unsafe_allow_html=True)
st.markdown(
    "<div class='disclaimer'>⚠️ <b>Disclaimer:</b> This tool is for research and educational purposes only. "
    "It is <b>not</b> a substitute for professional medical advice, diagnosis, or treatment.</div>",
    unsafe_allow_html=True,
)

default_text = st.session_state.pop("prefill", "")

col1, col2 = st.columns([1, 1.8], gap="large")

with col1:
    st.markdown("### 📝 Patient Consultation")
    question = st.text_area(
        label="Describe symptoms or paste a clinical case",
        value=default_text,
        height=200,
        placeholder="e.g. A 45-year-old male presents with chest pain radiating to the left arm …",
    )

    btn1, btn2 = st.columns(2)
    with btn1:
        analyze = st.button("🔬 Analyze Case", type="primary", use_container_width=True)
    with btn2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.pop("last_response", None)
            st.rerun()

with col2:
    st.markdown("### 🩺 Expert Reasoning & Diagnosis")
    response_placeholder = st.empty()

    if analyze:
        if not question.strip():
            st.warning("⚠️ Please enter a question before analyzing.")
        else:
            try:
                model, tokenizer = load_model()
            except Exception as e:
                st.error(f"❌ Failed to load model: {e}")
                st.stop()

            full_response = ""
            response_placeholder.markdown(
                "<div class='response-box'>⏳ Generating response …</div>",
                unsafe_allow_html=True,
            )
            for token in run_inference_stream(model, tokenizer, question):
                full_response += token
                response_placeholder.markdown(
                    f"<div class='response-box'>{full_response}</div>",
                    unsafe_allow_html=True,
                )
            st.session_state["last_response"] = full_response

    elif "last_response" in st.session_state:
        response_placeholder.markdown(
            f"<div class='response-box'>{st.session_state['last_response']}</div>",
            unsafe_allow_html=True,
        )
    else:
        response_placeholder.markdown(
            "<div class='response-box' style='color:#94a3b8;'>Response will appear here …</div>",
            unsafe_allow_html=True,
        )

st.markdown("""
<div class='footer'>
    Produced by <b>Priyanshu Sinha</b> &nbsp;·&nbsp;
    Fine-tuned on Medical-o1 Reasoning Dataset &nbsp;·&nbsp;
    Built with Unsloth &amp; Streamlit
</div>
""", unsafe_allow_html=True)
