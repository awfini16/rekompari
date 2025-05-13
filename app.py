import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.set_page_config(page_title="Rekomendasi Wisata", layout="centered")

st.title("🎯 Sistem Rekomendasi Wisata Berbasis TF-IDF dan Fitur Numerik")
st.write("Upload dataset dan masukkan nama wisata untuk mendapatkan rekomendasi.")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type="csv")

if uploaded_file is not None:
    # STEP 1: Load data
    df = pd.read_csv(uploaded_file, encoding='latin-1')

    # STEP 2: Preprocessing numerik
    df['Rating'] = df['Rating'].str.replace(',', '.').astype(float)
    df['Tiket Masuk Weekday'] = df['Tiket Masuk Weekday'].replace("Gratis", 0).astype(int)
    df['Tiket Masuk Weekend'] = df['Tiket Masuk Weekend'].replace("Gratis", 0).astype(int)

    # STEP 3: Normalisasi
    scaler = MinMaxScaler()
    df[['Rating_norm', 'Tiket_Weekday_norm', 'Tiket_Weekend_norm']] = scaler.fit_transform(
        df[['Rating', 'Tiket Masuk Weekday', 'Tiket Masuk Weekend']]
    )

    # STEP 4: TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['Deskripsi'].fillna(''))

    # STEP 5: Gabungkan fitur
    feature_matrix = np.hstack([
        tfidf_matrix.toarray(),
        df[['Rating_norm', 'Tiket_Weekday_norm', 'Tiket_Weekend_norm']].values
    ])

    # STEP 6: Cosine similarity
    cosine_sim = cosine_similarity(feature_matrix)

    # STEP 7: Fungsi rekomendasi
    def recommend(wisata_name, top_n=5):
        try:
            idx = df[df['Nama Wisata'].str.contains(wisata_name, case=False, na=False)].index[0]
        except IndexError:
            return None
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        recommended = df.iloc[[i[0] for i in sim_scores]][['Nama Wisata', 'Rating', 'Deskripsi']].copy()
        recommended['Similarity Score'] = [round(i[1], 3) for i in sim_scores]
        return recommended.reset_index(drop=True)

    # Input pengguna
    nama_wisata = st.text_input("Masukkan nama wisata (contoh: Papuma)")

    if nama_wisata:
        hasil = recommend(nama_wisata)
        if hasil is not None:
            st.subheader("📌 Rekomendasi Wisata:")
            st.dataframe(hasil)
        else:
            st.warning("Nama wisata tidak ditemukan. Coba kata kunci lain.")
else:
    st.info("Silakan upload file CSV terlebih dahulu.")
