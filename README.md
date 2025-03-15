# README

## 1️⃣ **Project: Word2Vec Skip-Gram for Language Translation**
### **📌 Deskripsi**
Proyek ini menggunakan model **Word2Vec Skip-Gram** untuk menghasilkan embedding kata dan membangun model penerjemahan berbasis neural network menggunakan LSTM. Dataset yang digunakan adalah kumpulan pasangan kata dari bahasa daerah (Toraja) ke Bahasa Indonesia.

### **📌 Fitur**
- **Word2Vec Skip-Gram:** Membangun embedding kata dengan cara memprediksi konteks dari kata target.
- **LSTM Encoder-Decoder:** Digunakan untuk menerjemahkan kalimat dari bahasa sumber ke bahasa target.
- **Pelatihan model dengan dataset bahasa daerah**
- **Evaluasi akurasi model**
- **Inferensi model untuk melakukan penerjemahan**

### **📌 Hasil & Evaluasi**
- Model dapat menerjemahkan bahasa Toraja ke Bahasa Indonesia dengan akurasi yang cukup baik setelah pelatihan.
- Evaluasi model dilakukan dengan membandingkan output model dengan terjemahan yang diharapkan.

---

## 2️⃣ **Project: LSTM Encoder-Decoder for Language Translation**
### **📌 Deskripsi**
Proyek ini menggunakan **LSTM Encoder-Decoder** untuk membangun sistem penerjemahan otomatis dari bahasa Toraja ke Bahasa Indonesia.

### **📌 Fitur**
- **Preprocessing data:** Tokenisasi dan padding data.
- **LSTM Encoder-Decoder:** Arsitektur yang digunakan untuk menangani urutan kata dalam terjemahan.
- **Beam Search Decoding:** Untuk meningkatkan akurasi hasil terjemahan.
- **Evaluasi model dengan metrik akurasi & visualisasi hasil pelatihan.**

### **📌 Hasil & Evaluasi**
- Model dilatih dengan dataset bahasa Toraja menggunakan 300 epoch.
- **Akurasi pelatihan dan validasi ditampilkan dalam bentuk grafik.**
- Hasil prediksi dibandingkan dengan **expected output** untuk melihat kualitas terjemahan.

---

## 3️⃣ **Project: Transformer-Based Translation Model**
### **📌 Deskripsi**
Proyek ini menggunakan arsitektur **Transformer** untuk membangun sistem penerjemahan bahasa berbasis Attention Mechanism.

### **📌 Fitur**
- **Self-Attention & Multi-Head Attention:** Untuk menangani hubungan kata dalam sebuah kalimat.
- **Positional Encoding:** Memastikan model memahami urutan kata.
- **Training & Evaluasi Model:** Menampilkan metrik akurasi dan loss.
- **Plot Training vs Validation Loss & Accuracy.**
- **Contoh terjemahan dengan perbandingan expected output & model output.**

### **📌 Hasil & Evaluasi**
- Model berhasil menerjemahkan bahasa Toraja ke Bahasa Indonesia.
- Visualisasi hasil pelatihan dengan grafik.
- Hasil prediksi dibandingkan dengan **expected output** untuk analisis performa model.

---

