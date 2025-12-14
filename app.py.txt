import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="COVID-19 CXR Detection", layout="centered")
st.title("CNN-Based COVID-19 Detection System (Chest X-ray)")
st.write("Upload a chest X-ray image. The model predicts: COVID or Normal.")
st.caption("⚠️ Educational use only — not a clinical diagnosis tool.")

# =========================
# SETTINGS
# =========================
MODEL_PATH = "covid_mobilenetv2_model.keras"  # must exist in the same repo
CLASS_NAMES = ["COVID", "Normal"]             # label 0, label 1 (same as training)
IMG_SIZE = (224, 224)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model file not found: {MODEL_PATH}\n\n"
            "✅ Put the model file inside the GitHub repo root (same folder as app.py) "
            "and name it exactly: covid_mobilenetv2_model.keras"
        )
        st.stop()

    # compile=False = أسرع + يقلل مشاكل التوافق
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()
st.success("✅ Model loaded")

st.caption(f"Class mapping: 0 → {CLASS_NAMES[0]} | 1 → {CLASS_NAMES[1]}")

# =========================
# UI
# =========================
threshold = st.slider("Decision threshold (for class 1 = Normal)", 0.10, 0.90, 0.50, 0.01)
show_probs = st.checkbox("Show probabilities", value=True)

uploaded = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])

# =========================
# HELPERS
# =========================
def preprocess(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(IMG_SIZE)
    x = np.array(pil_img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # (1,224,224,3)

    # ✅ IMPORTANT:
    # Do NOT divide by 255 here IF your model includes Rescaling(1./255) inside.
    # If you ever train a model without Rescaling, then do: x = x / 255.0
    return x

def predict(x: np.ndarray):
    p1 = float(model.predict(x, verbose=0)[0][0])  # P(label=1) = P(Normal)
    p0 = 1.0 - p1                                  # P(label=0) = P(COVID)
    pred_idx = 1 if p1 >= threshold else 0
    pred_name = CLASS_NAMES[pred_idx]
    conf = (p1 if pred_idx == 1 else p0) * 100.0
    return p0, p1, pred_idx, pred_name, conf

# =========================
# RUN PREDICTION
# =========================
if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    x = preprocess(img)
    p0, p1, pred_idx, pred_name, conf = predict(x)

    st.subheader("Prediction")
    st.write(f"Result: **{pred_name}**")
    st.write(f"Confidence: **{conf:.2f}%**")

    if show_probs:
        st.write("Probabilities")
        st.write(f"- P({CLASS_NAMES[0]}): {p0:.4f}")
        st.write(f"- P({CLASS_NAMES[1]}): {p1:.4f}")

    st.info("If outputs look biased, adjust the threshold slider and test multiple images.")
