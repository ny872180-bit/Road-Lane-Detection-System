# app.py

import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load Roberta for Detection
rob_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
rob_model = RobertaForSequenceClassification.from_pretrained("roberta-base")
rob_model.eval()

# Load T5 for Rewriting
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_model.eval()

@st.cache_data
def detect_text(text):
    inputs = rob_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = rob_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs).item()
    return prediction  # 0: Human, 1: AI/Plagiarized

def rewrite_text(text):
    input_text = f"paraphrase: {text}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = t5_model.generate(input_ids, max_length=256, num_beams=5, early_stopping=True)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def deplagiarize(text):
    is_plag = detect_text(text)
    if is_plag == 1:
        return rewrite_text(text)
    else:
        return "✅ This text appears to be human-written. No changes needed."

# ─────────────────────────────────────────────
# Streamlit UI
st.set_page_config(page_title="DePlagiarizer AI", layout="centered")

st.title("🧠 DePlagiarizer AI")
st.write("Detect and rewrite AI-generated or plagiarized research content to sound more human and original.")

input_text = st.text_area("✍️ Paste your paragraph below:", height=200)

if st.button("Deplagiarize"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing and rewriting..."):
            result = deplagiarize(input_text)
            st.subheader("📝 Output:")
            st.success(result)
