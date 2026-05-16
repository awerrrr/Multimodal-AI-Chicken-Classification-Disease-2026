import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle

from PIL import Image

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Deteksi Penyakit Ayam",
    page_icon="🐔",
    layout="centered"
)

# ============================================================
# CONSTANT
# ============================================================
IMG_SIZE = 224
MAX_LEN = 150
MAX_SYMPTOMS = 5

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
    color: white;
}

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 4rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 2rem;
}

.hero-box {
    background: linear-gradient(135deg, #1e293b, #111827);
    padding: 30px;
    border-radius: 25px;
    border: 1px solid #334155;
    margin-bottom: 35px;
}

.result-card {
    background: #111827;
    border-radius: 25px;
    padding: 25px;
    border: 1px solid #374151;
    margin-top: 20px;
}

.metric-card {
    background: #1e293b;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid #334155;
}

.metric-title {
    color: #94a3b8;
    font-size: 14px;
}

[data-testid="stMetric"] {
    background-color: #1e293b;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_components():

    # LOAD MODEL
    model = tf.keras.models.load_model(
        "model/best_multimodal_macro_ncd.h5",
        compile=False
    )

    # LOAD TOKENIZER
    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # LOAD LABEL ENCODER
    with open("model/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return model, tokenizer, le

# ============================================================
# LOAD SEMUA KOMPONEN
# ============================================================
model, tokenizer, le = load_components()

# ============================================================
# PREPROCESS IMAGE
# ============================================================
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# ============================================================
# PREPROCESS TEXT
# ============================================================
def preprocess_text(text):

    if text.strip() == "":
        text = "ayam sehat"

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding='post'
    )

    return padded.astype(np.int32)

# ============================================================
# FUNGSI PREDIKSI
# ============================================================
def predict_model(img_array, text_array):

    prediction = model.predict(
        [img_array, text_array],
        verbose=0
    )

    return prediction[0]

# ============================================================
# LABEL DISPLAY & INFO
# ============================================================
DISPLAY_LABELS = {
    "coccidiosis": "Coccidiosis",
    "healthy": "Healthy",
    "new_castle_disease": "New Castle Disease",
    "salmonella": "Salmonella"
}

DISEASE_INFO = {
    "coccidiosis": """
    **Coccidiosis** adalah penyakit parasit yang menyerang saluran pencernaan ayam.

    Gejala umum:
    - Diare berdarah
    - Tubuh lemah
    - Penurunan berat badan
    - Nafsu makan menurun
    """,

    "healthy": """
    **Healthy** menunjukkan ayam berada dalam kondisi sehat.

    Indikator umum:
    - Gerakan aktif
    - Nafsu makan baik
    - Bulu bersih
    - Tidak menunjukkan gejala penyakit
    """,

    "new_castle_disease": """
    **New Castle Disease** adalah penyakit virus yang sangat menular pada unggas.

    Gejala umum:
    - Gangguan pernapasan
    - Leher atau kepala terpuntir
    - Kelumpuhan
    - Tremor atau gangguan saraf
    """,

    "salmonella": """
    **Salmonella** adalah infeksi bakteri yang dapat menyerang sistem pencernaan ayam.

    Gejala umum:
    - Diare
    - Tubuh lemah
    - Nafsu makan menurun
    - Penurunan kondisi tubuh
    """
}


# ============================================================
# SYMPTOM OPTIONS
# ============================================================
SYMPTOM_GROUPS = {
    "Kondisi Normal": [
        "ayam aktif",
        "nafsu makan baik",
        "bulu bersih",
        "kondisi tubuh normal",
        "tidak menunjukkan gejala penyakit",
    ],
    "Gangguan Pencernaan": [
        "diare",
        "diare berdarah",
        "feses merah",
        "feses hijau",
        "feses tidak normal",
        "penurunan berat badan",
    ],
    "Gangguan Pernapasan": [
        "gangguan pernapasan",
        "ngorok",
        "sesak napas",
        "batuk",
        "lendir hidung",
        "mata berair",
    ],
    "Gangguan Saraf / NCD": [
        "kepala terpuntir",
        "leher memutar",
        "tortikolis",
        "tremor",
        "kejang",
        "kelumpuhan kaki",
        "kelumpuhan sayap",
        "kehilangan keseimbangan",
    ],
    "Kondisi Tubuh & Perilaku": [
        "ayam lemah",
        "ayam lesu",
        "tidak aktif",
        "nafsu makan menurun",
        "penurunan aktivitas",
        "demam",
        "bulu kusut",
    ],
}

DEFAULT_SYMPTOM_TEXT = "ayam sehat kondisi tubuh normal nafsu makan baik ayam aktif"


def build_symptom_text(selected_symptoms):
    if not selected_symptoms:
        return DEFAULT_SYMPTOM_TEXT
    return " ".join(selected_symptoms)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero-box">
    <div class="title">Chicken Disease Detection AI</div>
    <div class="subtitle">
        Sistem deteksi penyakit ayam berbasis CNN dan NLP menggunakan Multimodal Deep Learning.
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Tentang Dashboard")

    st.write("""
    Dashboard ini digunakan untuk membantu klasifikasi penyakit ayam berdasarkan:

    - Citra feses ayam
    - Teks gejala klinis
    - Model multimodal CNN + NLP
    """)

    st.markdown("**Kelas klasifikasi:**")
    st.write("""
    - Coccidiosis
    - Healthy
    - New Castle Disease
    - Salmonella
    """)

    st.info("Hasil prediksi digunakan sebagai bantuan analisis, bukan pengganti diagnosis dokter hewan.")

# ============================================================
# INPUT SECTION
# ============================================================
st.subheader("Input Data")

input_col1, input_col2 = st.columns([1.1, 1])

with input_col1:
    uploaded_file = st.file_uploader(
        "Upload gambar ayam *",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(
            uploaded_file,
            caption="Gambar yang diupload",
            use_container_width=True
        )

with input_col2:
    st.markdown("**Pilih gejala ayam**")
    st.caption(f"Pilih maksimal {MAX_SYMPTOMS} gejala utama yang paling terlihat. ")

    selected_symptoms = []

    for group_name, symptoms in SYMPTOM_GROUPS.items():
        with st.expander(group_name, expanded=(group_name in ["Gangguan Saraf / NCD", "Kondisi Tubuh & Perilaku"])):
            check_col1, check_col2 = st.columns(2)

            for idx, symptom in enumerate(symptoms):
                key = f"symptom_{group_name}_{idx}"
                current_value = st.session_state.get(key, False)
                limit_reached = len(selected_symptoms) >= MAX_SYMPTOMS
                disabled = limit_reached and not current_value
                target_col = check_col1 if idx % 2 == 0 else check_col2

                with target_col:
                    checked = st.checkbox(
                        symptom.title(),
                        key=key,
                        disabled=disabled
                    )

                    if checked:
                        selected_symptoms.append(symptom)

    symptom_text = build_symptom_text(selected_symptoms)

    if selected_symptoms:
        st.success(f"{len(selected_symptoms)} dari {MAX_SYMPTOMS} gejala dipilih.")
        if len(selected_symptoms) >= MAX_SYMPTOMS:
            st.warning("Batas pilihan gejala tercapai. Hapus salah satu pilihan untuk memilih gejala lain.")
        with st.expander("Lihat teks gejala yang dikirim ke model"):
            st.write(symptom_text)
    else:
        st.info("Belum ada gejala dipilih. Sistem akan memakai kondisi default ayam sehat.")

# ============================================================
# PREDICTION BUTTON
# ============================================================
st.markdown("---")

predict_clicked = st.button(
    "Prediksi Penyakit",
    use_container_width=True
)

# ============================================================
# PREDICTION RESULT
# ============================================================
if predict_clicked:

    if uploaded_file is None:
        st.warning("Silakan upload gambar ayam terlebih dahulu.")
        st.stop()

    with st.spinner("Sedang menganalisis gambar dan gejala..."):

        try:
            img_array = preprocess_image(uploaded_file)
            text_array = preprocess_text(symptom_text)

            # Multimodal prediction
            multimodal_probs = predict_model(
                img_array,
                text_array
            )

            # Image-only approximation
            dummy_text = np.zeros(
                (1, MAX_LEN),
                dtype=np.int32
            )

            image_only_probs = predict_model(
                img_array,
                dummy_text
            )

            # Text-only approximation
            dummy_image = np.zeros(
                (1, IMG_SIZE, IMG_SIZE, 3),
                dtype=np.float32
            )

            text_only_probs = predict_model(
                dummy_image,
                text_array
            )

            pred_idx = np.argmax(multimodal_probs)

            pred_label = le.inverse_transform(
                [pred_idx]
            )[0]

            pred_label_display = DISPLAY_LABELS.get(
                pred_label,
                pred_label
            )

            confidence = float(np.max(multimodal_probs)) * 100

            # ====================================================
            # MAIN RESULT
            # ====================================================
            result_html = f"""
<div class="result-card">
  <p style="color:#94a3b8; font-size:15px; margin:0 0 6px 0;">Hasil Deteksi Penyakit</p>
  <h1 style="color:#22c55e; font-size:42px; margin:0 0 14px 0;">{pred_label_display}</h1>
  <p style="color:#cbd5e1; font-size:17px; margin:0 0 8px 0;">
    Sistem memperkirakan ayam mengalami kondisi tersebut dengan tingkat keyakinan:
  </p>
  <h2 style="color:#38bdf8; margin:0;">{confidence:.2f}%</h2>
</div>
"""
            st.markdown(result_html, unsafe_allow_html=True)

            # ====================================================
            # AI INSIGHT
            # ====================================================
            if confidence > 90:
                st.success("Model sangat yakin terhadap hasil prediksi.")
            elif confidence > 70:
                st.info("Model cukup yakin terhadap hasil prediksi.")
            else:
                st.warning("Confidence rendah. Disarankan pemeriksaan manual.")

            # ====================================================
            # PROBABILITY DISTRIBUTION
            # ====================================================
            st.subheader("Probabilitas Setiap Kelas")

            for i, class_name in enumerate(le.classes_):
                prob = float(multimodal_probs[i]) * 100
                label_display = DISPLAY_LABELS.get(class_name, class_name)

                st.write(f"**{label_display}** — {prob:.2f}%")
                st.progress(prob / 100)

            # ====================================================
            # MODALITY COMPARISON
            # ====================================================
            st.subheader("Perbandingan Confidence Modalitas")

            multi_conf = float(np.max(multimodal_probs)) * 100
            image_conf = float(np.max(image_only_probs)) * 100
            text_conf = float(np.max(text_only_probs)) * 100

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Multimodal", f"{multi_conf:.1f}%")

            with metric_col2:
                st.metric("CNN Image", f"{image_conf:.1f}%")

            with metric_col3:
                st.metric("NLP Text", f"{text_conf:.1f}%")

            st.caption(
                "Catatan: image-only dan text-only di sini adalah simulasi dengan input dummy, "
                "bukan model terpisah yang dilatih secara independen."
            )

            # ====================================================
            # DISEASE INFO
            # ====================================================
            with st.expander("Informasi Penyakit", expanded=True):
                st.markdown(
                    DISEASE_INFO.get(
                        pred_label,
                        "Informasi penyakit tidak tersedia."
                    )
                )

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {str(e)}")
# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">

🐔 Chicken Disease Detection System
<br>
Multimodal Deep Learning Research Project
<br><br>
TensorFlow • Streamlit • CNN • NLP

</div>
""", unsafe_allow_html=True)
