# 🐔 Multimodal AI for Chicken Disease Classification (2026)

## 📌 Overview

Project ini bertujuan untuk mengembangkan sistem berbasis **Multimodal Artificial Intelligence** yang mampu mengidentifikasi penyakit ayam secara otomatis menggunakan kombinasi data **citra (image)** dan **informasi tekstual/gejala**.
Dalam guna untuk memenuhi kriteria kelulusan *Sarjana Muda Sistem Informasi Universitas Gunadarma 2026*.

Solusi ini dirancang untuk membantu peternak dalam:

* Deteksi dini penyakit ayam
* Mengurangi risiko kematian ternak
* Meningkatkan efisiensi pengambilan keputusan

---

## 🚀 Key Features

* 🧠 **Image Classification (CNN)** untuk mendeteksi penyakit dari gambar ayam
* 📝 **Text-based Information Extraction** untuk memahami gejala penyakit
* 🔗 **Multimodal Integration** (Image + Text) untuk meningkatkan akurasi prediksi
* 📊 Evaluasi performa model (Classification Report, Confusion Matrix, etc.)
* ⚙️ End-to-end pipeline dari preprocessing hingga inference

---

## 🧩 Problem Statement

Deteksi penyakit ayam secara manual memiliki beberapa kendala:

* Membutuhkan pengalaman tinggi
* Rentan terhadap kesalahan manusia
* Tidak scalable untuk peternakan besar

Project ini mencoba menjawab masalah tersebut dengan pendekatan AI berbasis data.

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras**
* **CNN (Convolutional Neural Network)**
* **Natural Language Processing (NLP)**
* **Scikit-learn**
* **Pandas & NumPy**

---

## 📂 Dataset

Dataset terdiri dari:

* 📷 Gambar ayam berdasarkan kategori penyakit.
Data gambar dapat diunduh pada [[Data Gambar Ayam:](https://www.kaggle.com/datasets/efoeetienneblavo/chicken-disease-dataset)]
* 📝 Data teks (gejala penyakit)

Struktur dataset:

```
dataset/
├── train/
├── validation/
└── test/
```

---

## 🔄 Pipeline

1. **Data Collection**
   - Dataset gambar ayam diperoleh dari Kaggle.
   - Dataset teks gejala/NLP disusun dan diolah terlebih dahulu di luar pipeline utama.

2. **Data Preparation**
   - Data gambar dibagi ke dalam folder `train`, `validation`, dan `test`.
   - Label gambar dinormalisasi agar konsisten dengan label model:
     - `coccidiosis`
     - `healthy`
     - `new_castle_disease`
     - `salmonella`
   - Dataset NLP yang sudah diproses dimuat dari file CSV.
   - Teks gejala dipasangkan ke data gambar berdasarkan label penyakit.

3. **Text Preprocessing**
   - Membersihkan teks dari nama label penyakit untuk mengurangi data leakage.
   - Normalisasi istilah medis dan gejala.
   - Penambahan variasi gejala khusus New Castle Disease, seperti:
     - kepala terpuntir
     - tortikolis
     - gangguan saraf
     - tremor
     - kelumpuhan kaki/sayap
   - Tokenisasi teks menggunakan Keras Tokenizer.
   - Padding sequence dengan panjang maksimum 150 token.

4. **Image Preprocessing**
   - Gambar dibaca menggunakan OpenCV.
   - Konversi warna dari BGR ke RGB.
   - Resize gambar menjadi `224 x 224`.
   - Normalisasi menggunakan `preprocess_input` dari MobileNetV2.
   - Augmentasi gambar diterapkan pada data training.

5. **Data Balancing**
   - Oversampling dilakukan pada data training agar jumlah sampel tiap kelas seimbang.
   - Kelas New Castle Disease yang memiliki jumlah data lebih sedikit diseimbangkan dengan kelas lain.

6. **Model Development**
   - Cabang image menggunakan MobileNetV2 sebagai feature extractor.
   - Cabang teks menggunakan Embedding dan Bidirectional LSTM.
   - Kedua fitur digabungkan menggunakan multimodal fusion.
   - Output model berupa klasifikasi 4 kelas penyakit ayam.

7. **Training & Fine-tuning**
   - Training awal dilakukan dengan base model MobileNetV2 dalam kondisi frozen.
   - Fine-tuning dilakukan dengan membuka sebagian layer akhir MobileNetV2.
   - Checkpoint model disimpan berdasarkan performa terbaik.

8. **Model Evaluation**
   - Evaluasi dilakukan menggunakan:
     - accuracy
     - precision
     - recall
     - F1-score
     - classification report
     - confusion matrix
   - Evaluasi juga membandingkan performa:
     - multimodal
     - image-only
     - text-only

9. **Deployment**
   - Model terbaik disimpan dan digunakan pada dashboard Streamlit.
   - Dashboard menerima input berupa gambar ayam dan pilihan gejala.
   - Input gejala dibuat dalam bentuk checkbox untuk mengurangi risiko typo dari pengguna.
---

## 📊 Results

* Model CNN berhasil mengklasifikasikan penyakit ayam dengan performa yang baik
* Integrasi multimodal menunjukkan potensi peningkatan akurasi dibanding single model
* Model mampu menangkap pola visual dan gejala secara bersamaan
* Model berhasil mendapatkan akurasi sebesar **98,9%**

---

## 💡 Business Impact

* 🐓 Membantu peternak mendeteksi penyakit lebih cepat
* 📉 Mengurangi kerugian akibat kematian ternak
* 📈 Meningkatkan efisiensi operasional peternakan
* 🤖 Langkah awal menuju **Smart Farming berbasis AI**

---

## 📌 Future Improvements

* Deployment ke aplikasi web/mobile
* Penambahan dataset untuk meningkatkan generalisasi model
* Optimasi model multimodal
* Integrasi dengan sistem IoT peternakan

---

## 🤝 Contribution

Terbuka untuk kolaborasi dan pengembangan lebih lanjut.
Silakan fork repository ini atau hubungi saya untuk diskusi.

---

## 📬 Contact

📧 Email: [najwar105@gmail.com](mailto:najwar105@gmail.com)
🔗 LinkedIn: linkedin.com/in/najwarputra

---

💡 *"Applying AI to solve real-world problems in agriculture."*
