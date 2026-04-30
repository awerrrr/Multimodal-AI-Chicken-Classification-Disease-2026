import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Deteksi Penyakit Ayam",
    layout="wide"
)

IMG_SIZE = (224, 224)

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("multimodal_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return model, tokenizer, le

model, tokenizer, le = load_model()

# ======================
# PREPROCESS
# ======================
def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def process_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    return padded

# ======================
# SIDEBAR
# ======================
st.sidebar.title("⚙️ Input Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload Gambar Ayam",
    type=["jpg", "png", "jpeg"]
)

text_input = st.sidebar.text_area(
    "Masukkan Gejala",
    placeholder="contoh: ayam lesu, nafsu makan turun..."
)

predict_btn = st.sidebar.button("🔍 Prediksi")

# ======================
# MAIN UI
# ======================
st.title("🐔 Sistem Deteksi Penyakit Ayam Berbasis Multimodal AI")

col1, col2 = st.columns(2)

# ======================
# PREVIEW
# ======================
with col1:
    st.subheader("📷 Gambar Input")
    if uploaded_file:
        st.image(uploaded_file, width="stretch")
    else:
        st.info("Upload gambar untuk melihat preview")

# ======================
# HASIL
# ======================
with col2:
    st.subheader("📊 Hasil Prediksi")

    if predict_btn:
        if uploaded_file is not None and text_input != "":
            
            img = load_image(uploaded_file)
            txt = process_text(text_input)

            pred = model.predict([img, txt])
            class_idx = np.argmax(pred)
            confidence = np.max(pred)

            label = le.inverse_transform([class_idx])[0]

            # 🔥 hasil utama
            st.success(f"Prediksi: {label}")
            st.progress(float(confidence))

            st.write(f"Confidence: **{confidence:.2f}**")

            # 🔥 semua probabilitas
            st.markdown("### 🔍 Detail Probabilitas")

            for i, prob in enumerate(pred[0]):
                st.write(f"{le.classes_[i]} : {prob:.2f}")

        else:
            st.warning("Masukkan gambar dan gejala terlebih dahulu")

# ======================
# FOOTER
# ======================
st.markdown("---")
st.caption("Multimodal Deep Learning | Image + Text | CNN + LSTM")
