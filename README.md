# **README**

## **📌 Proyek 1: Word2Vec Skip-Gram for Text Analysis**
### **Deskripsi**
Proyek ini menggunakan model **Word2Vec (Skip-Gram)** untuk menganalisis hubungan antar kata dalam sebuah korpus teks.
Model ini dilatih menggunakan dataset teks yang telah diproses untuk membangun embedding kata, yang kemudian divisualisasikan dengan **t-SNE** dan **PCA**. Selain itu, model dapat digunakan untuk menemukan kata-kata yang memiliki makna serupa berdasarkan **cosine similarity**.

### **Fitur Utama**
- **Preprocessing teks:** Tokenisasi, penghapusan tanda baca, dan stopwords.
- **Pelatihan Word2Vec dengan Skip-Gram.**
- **Visualisasi embedding menggunakan PCA dan t-SNE.**
- **Analisis hubungan antar kata dengan cosine similarity.**

### **Contoh Kode**
```python
from gensim.models import Word2Vec
word2vec_sg = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, sg=1)
vector_sg = word2vec_sg.wv['kata']
```

---

## **📌 Proyek 2: LSTM Encoder-Decoder for Language Translation**
### **Deskripsi**
Proyek ini menggunakan model **Encoder-Decoder berbasis LSTM** untuk melakukan penerjemahan dari bahasa daerah ke Bahasa Indonesia dengan teknik **sequence-to-sequence (seq2seq)**.

### **Fitur Utama**
- **Preprocessing Data**: Tokenisasi teks, padding sequence, pembuatan indeks kata.
- **LSTM Encoder-Decoder**: Model neural network untuk menerjemahkan kalimat.
- **Inferensi dengan Beam Search untuk meningkatkan kualitas terjemahan.**
- **Evaluasi Model dengan Akurasi & Visualisasi Loss.**

### **Contoh Kode**
```python
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
```

---

## **📌 Proyek 3: Transformer-Based Language Translation**
### **Deskripsi**
Proyek ini menggunakan **Transformer Model** untuk melakukan penerjemahan dari bahasa daerah ke Bahasa Indonesia. Model ini didasarkan pada arsitektur **Attention is All You Need**.

### **Fitur Utama**
- **Self-Attention & Multi-Head Attention** untuk menangani hubungan antar kata.
- **Positional Encoding** untuk mempertahankan urutan kata dalam kalimat.
- **Training & Evaluasi Model** dengan metrik akurasi dan loss.
- **Inferensi Model untuk Penerjemahan.**

### **Contoh Kode**
```python
def build_transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
    inputs = tf.keras.Input(shape=(None,))
    targets = tf.keras.Input(shape=(None,))
    enc_embedding = Embedding(input_vocab_size, d_model)(inputs)
    dec_embedding = Embedding(target_vocab_size, d_model)(targets)
    return tf.keras.Model(inputs=[inputs, targets], outputs=final_output)
```

---

## **📌 Catatan Umum**
- **Word2Vec Skip-Gram lebih cocok untuk analisis hubungan antar kata,** sedangkan **LSTM dan Transformer digunakan untuk penerjemahan bahasa.**
- **Gunakan GPU untuk mempercepat pelatihan model.**
- **Transformer lebih kompleks dibandingkan LSTM, tetapi memiliki performa lebih baik untuk penerjemahan teks.**

🚀 **Selamat Mencoba!** 🚀

