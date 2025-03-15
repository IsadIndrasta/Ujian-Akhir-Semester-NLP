# README

## 1️⃣ **Project: Word2Vec with Skip-Gram for Text Analysis**
📌 Deskripsi Proyek

Proyek ini menggunakan model Word2Vec (Skip-Gram) untuk menganalisis hubungan antar kata dalam sebuah korpus teks.
Model ini dilatih menggunakan dataset teks yang telah diproses untuk membangun embedding kata, yang kemudian divisualisasikan dengan t-SNE dan PCA. Selain itu, model dapat digunakan untuk menemukan kata-kata yang memiliki makna serupa berdasarkan cosine similarity.

📌 Fitur Utama

Preprocessing teks:

Konversi ke huruf kecil.

Penghapusan tanda baca dan stopwords.

Tokenisasi teks.

Pembuatan pasangan konteks menggunakan Skip-Gram.

Pelatihan model Word2Vec dari awal menggunakan numpy.

Visualisasi embedding dengan PCA dan t-SNE.

Analisis hubungan antar kata dengan cosine similarity.

📌 Instalasi

Pastikan dependensi berikut sudah terpasang sebelum menjalankan kode:

pip install numpy pandas nltk scikit-learn matplotlib

📌 Cara Menjalankan

Lakukan Preprocessing Teks:

Tokenisasi teks dan hilangkan stopwords.

Buat pasangan Skip-Gram untuk melatih model Word2Vec.

Latih model Word2Vec dengan Skip-Gram.

Visualisasikan Word Embeddings menggunakan PCA & t-SNE.

Lakukan analisis hubungan antar kata dengan cosine similarity.

📌 Contoh Kode

Pelatihan Skip-Gram Word2Vec

from gensim.models import Word2Vec

# Latih model Word2Vec dengan Skip-Gram
word2vec_sg = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, sg=1)

# Ambil vektor kata dari model
vector_sg = word2vec_sg.wv['kata']

Visualisasi dengan PCA

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduksi dimensi dengan PCA
pca = PCA(n_components=2)
word_vecs_2d = pca.fit_transform(W1)

# Plot hasil PCA
plt.figure(figsize=(10,6))
plt.scatter(word_vecs_2d[:, 0], word_vecs_2d[:, 1], marker='o', edgecolors='k', s=100)

for i, word in enumerate(vocab[:50]):
    plt.annotate(word, xy=(word_vecs_2d[i, 0], word_vecs_2d[i, 1]), fontsize=12)

plt.title("Visualisasi Word Embedding dengan PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()

Analisis Hubungan Antar Kata

def most_similar(word, top_n=5):
    if word not in word_to_index:
        return []
    word_idx = word_to_index[word]
    word_vector = W1[word_idx]
    similarities = cosine_similarity([word_vector], W1)[0]
    similar_indices = np.argsort(similarities)[::-1][1:top_n + 1]
    return [(index_to_word[i], similarities[i]) for i in similar_indices]

📌 Hasil & Evaluasi

Model berhasil mengelompokkan kata-kata yang memiliki hubungan semantik.

t-SNE digunakan untuk memproyeksikan word embeddings ke ruang 2D.

PCA digunakan untuk memahami distribusi kata dalam vektor berdimensi lebih rendah.

Cosine similarity digunakan untuk mencari kata-kata dengan makna yang mirip.
---

## 2️⃣ **Project: Encoder-Decoder for Language Translation**
📌 Deskripsi Proyek

Proyek ini menggunakan model Encoder-Decoder berbasis LSTM untuk melakukan penerjemahan dari bahasa daerah ke Bahasa Indonesia. Model ini dilatih menggunakan dataset pasangan kalimat dan menggunakan teknik sequence-to-sequence (seq2seq) untuk menghasilkan output terjemahan yang lebih akurat.

📌 Fitur Utama

Preprocessing Data: Tokenisasi teks, padding sequence, dan pembuatan indeks kata.

LSTM Encoder-Decoder: Model neural network untuk menerjemahkan kalimat.

Dropout & Regularisasi: Untuk meningkatkan generalisasi model.

Evaluasi Model dengan Akurasi & Visualisasi Loss.

Inferensi dengan Beam Search untuk meningkatkan kualitas terjemahan.

📌 Instalasi

Pastikan dependensi berikut sudah terpasang sebelum menjalankan kode:

pip install tensorflow numpy pandas matplotlib

📌 Cara Menjalankan

Muat Dataset: Dataset berisi pasangan kalimat bahasa daerah dan bahasa Indonesia.

Preprocessing: Tokenisasi teks, konversi ke indeks, padding sequences.

Latih Model Encoder-Decoder dengan LSTM.

Evaluasi Model: Plot grafik loss dan akurasi.

Gunakan Model untuk Inferensi: Masukkan kalimat bahasa daerah dan lihat hasil terjemahan.

📌 Contoh Kode

Latih Model LSTM Encoder-Decoder

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

# Parameter Model
latent_dim = 256
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Kompilasi Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

Evaluasi Model & Visualisasi Loss

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

Inferensi Model untuk Penerjemahan

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['<start>']] = 1.0
    decoded_sentence = ''

    for _ in range(max_decoder_seq_length):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == '<end>' or len(decoded_sentence) > max_decoder_seq_length:
            break

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]

    return decoded_sentence

📌 Hasil & Evaluasi

Model berhasil menerjemahkan bahasa daerah ke Bahasa Indonesia dengan akurasi yang cukup baik.

Plot loss menunjukkan konvergensi model selama pelatihan.

Inferensi model bekerja dengan baik dalam menghasilkan terjemahan yang sesuai.

---

## 3️⃣ **Project: Transformer-Based Language Translation**
📌 Deskripsi Proyek

Proyek ini menggunakan Transformer Model untuk melakukan penerjemahan dari bahasa daerah ke Bahasa Indonesia. Model ini didasarkan pada arsitektur Attention is All You Need dan menggunakan Self-Attention, Multi-Head Attention, dan Positional Encoding untuk menangani pemrosesan sekuensial teks.

📌 Fitur Utama

Preprocessing Data: Tokenisasi teks, padding sequence, dan pembuatan indeks kata.

Transformer Model: Implementasi Self-Attention dan Multi-Head Attention.

Positional Encoding: Menangani urutan kata dalam kalimat.

Training & Evaluasi Model: Plot metrik akurasi dan loss.

Inferensi Model untuk Penerjemahan.

📌 Instalasi

Pastikan dependensi berikut sudah terpasang sebelum menjalankan kode:

pip install tensorflow numpy pandas matplotlib

📌 Cara Menjalankan

Muat Dataset: Dataset berisi pasangan kalimat bahasa daerah dan bahasa Indonesia.

Preprocessing: Tokenisasi teks, konversi ke indeks, padding sequences.

Latih Model Transformer.

Evaluasi Model: Plot grafik loss dan akurasi.

Gunakan Model untuk Inferensi: Masukkan kalimat bahasa daerah dan lihat hasil terjemahan.

📌 Contoh Kode

Bangun Model Transformer

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout

def build_transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
    inputs = tf.keras.Input(shape=(None,))
    targets = tf.keras.Input(shape=(None,))
    enc_embedding = Embedding(input_vocab_size, d_model)(inputs)
    dec_embedding = Embedding(target_vocab_size, d_model)(targets)
    encoder = enc_embedding
    for _ in range(num_layers):
        encoder = TransformerBlock(d_model, num_heads, dff, rate)(encoder)
    decoder = dec_embedding
    for _ in range(num_layers):
        decoder = TransformerBlock(d_model, num_heads, dff, rate)(decoder)
    final_output = Dense(target_vocab_size, activation='softmax')(decoder)
    return tf.keras.Model(inputs=[inputs, targets], outputs=final_output)

Evaluasi Model & Visualisasi Loss

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

Inferensi Model untuk Penerjemahan

def translate_sentence(input_sentence):
    input_seq = tokenizer.texts_to_sequences([input_sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_length, padding='post')
    predicted_seq = transformer.predict([input_seq])
    translated_sentence = tokenizer.sequences_to_texts(predicted_seq.argmax(axis=-1))
    return translated_sentence

📌 Hasil & Evaluasi

Model berhasil menerjemahkan bahasa daerah ke Bahasa Indonesia dengan hasil yang cukup baik.

Plot loss menunjukkan konvergensi model selama pelatihan.

Inferensi model dapat menghasilkan terjemahan dengan lebih akurat dibandingkan metode LSTM Encoder-Decoder.

---

