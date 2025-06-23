import os
os.environ["USE_TF"] = "0"

import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime
import time

# ------------------- Model Loader -------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")  # ⬅️ Lighter & faster
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

# ------------------- Optimized Answer Generator -------------------
def generate_answer(tokenizer, model, context, question):
    input_text = f"question: {question} context: {context[-1000:]}"  # Trim context for speed
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,         # Shorter response = faster
        num_beams=2,               # Good quality with low latency
        do_sample=False,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------- 🌸 Styling -------------------
st.markdown("""
<style>
body {
    background-color: #fef6fd;
    font-family: 'Comic Sans MS', 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #a64ac9;
}
.chat-container {
    background-color: #fff0f6;
    border-radius: 20px;
    padding: 1.5rem;
    max-width: 750px;
    margin: 2rem auto 6rem auto;
    box-shadow: 0 4px 20px rgba(255, 182, 193, 0.3);
    font-size: 1rem;
}
.chat-bubble {
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    border-radius: 18px;
    max-width: 85%;
    line-height: 1.5;
}
.user-bubble {
    background-color: #c5dfff;
    color: #333;
    margin-left: auto;
    text-align: right;
}
.bot-bubble {
    background-color: #ffe0f0;
    color: #222;
    margin-right: auto;
    text-align: left;
}
.message-meta {
    font-size: 0.7rem;
    color: #888;
    margin-top: 0.3rem;
}
.input-box {
    position: fixed;
    bottom: 20px;
    left: 0;
    right: 0;
    max-width: 750px;
    margin: auto;
    padding: 0 1rem;
}
input[type="text"] {
    background-color: #fff;
    border: 2px solid #f9d5ec;
    border-radius: 16px;
    padding: 1rem;
    width: 100%;
    font-size: 1rem;
    color: #000;
}
</style>
""", unsafe_allow_html=True)

# ------------------- 🐱 Header -------------------
st.markdown("<h1 style='text-align:center;'>💬 PDF Chat Café</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: #999;'>Upload a PDF and chat with it like it's your study buddy ☕</p>", unsafe_allow_html=True)

# ------------------- 📎 Upload PDF -------------------
uploaded_file = st.file_uploader("📎 Upload your adorable PDF", type="pdf", label_visibility="collapsed")

if uploaded_file and "context" not in st.session_state:
    with st.spinner("🌸 Reading your precious PDF..."):
        st.session_state.context = extract_text_from_pdf(uploaded_file)
        st.session_state.tokenizer, st.session_state.model = load_model()
        st.session_state.chat_history = [(
            "assistant", f"Hiya! I'm ready to chat with your file **{uploaded_file.name}**! Ask me anything 💕", datetime.now()
        )]
    st.session_state.uploaded_file = uploaded_file

# ------------------- 💬 Chat Interface -------------------
if "context" in st.session_state:
    file_name = st.session_state.uploaded_file.name
    st.markdown(f"""
        <div class='chat-container'>
            <p style='font-weight:bold; color:#000;'>📁 You're chatting with: {file_name}</p>
    """, unsafe_allow_html=True)

    for sender, msg, timestamp in st.session_state.chat_history:
        role_class = "user-bubble" if sender == "user" else "bot-bubble"
        emoji = "🧑‍🎓" if sender == "user" else "🤖"
        st.markdown(f"""
            <div class='chat-bubble {role_class}'>
                <b>{emoji} {sender.title()}:</b><br>{msg}
                <div class='message-meta'>{timestamp.strftime('%I:%M %p')}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------- 📬 Message Input Form -------------------
    with st.form("message_form", clear_on_submit=True):
        st.markdown("<div class='input-box'>", unsafe_allow_html=True)
        user_query = st.text_input("Ask a question about your PDF... ✨", label_visibility="collapsed")
        submitted = st.form_submit_button("📨 Send")
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted and user_query.strip():
        st.session_state.chat_history.append(("user", user_query, datetime.now()))
        with st.spinner("🎀 Thinking... hold on a sparkle!"):
            start = time.time()
            answer = generate_answer(
                st.session_state.tokenizer,
                st.session_state.model,
                st.session_state.context,
                user_query
            )
            while time.time() - start < 1.2:
                time.sleep(0.1)  # ✨ Just enough sparkle, no delay bloat
            st.session_state.chat_history.append(("assistant", answer, datetime.now()))
        st.rerun()
else:
    st.info("🌼 Please upload your adorable PDF to begin chatting.")
