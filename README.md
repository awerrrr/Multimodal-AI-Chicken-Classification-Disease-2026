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

1. Data Collection
2. Data Preprocessing

   * Image resizing & normalization
   * Text cleaning & extraction
3. Model Development

   * CNN untuk image classification
   * NLP untuk ekstraksi informasi
4. Multimodal Fusion
5. Model Evaluation
6. Deployment-ready model

---

## 📊 Results

* Model CNN berhasil mengklasifikasikan penyakit ayam dengan performa yang baik
* Integrasi multimodal menunjukkan potensi peningkatan akurasi dibanding single model
* Model mampu menangkap pola visual dan gejala secara bersamaan
* Model berhasil mendapatkan akurasi sebesar **93%**

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
