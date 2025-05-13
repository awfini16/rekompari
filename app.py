import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# LOAD DAN CLEAN DATASET
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("datasetku.csv", encoding='latin-1')

    # Cleaning data
    df['Deskripsi'] = df['Deskripsi'].fillna('')
    df['Rating'] = df['Rating'].str.replace(',', '.').astype(float)
    df['Tiket Masuk Weekday'] = pd.to_numeric(df['Tiket Masuk Weekday'], errors='coerce').fillna(0)
    df['Tiket Masuk Weekend'] = pd.to_numeric(df['Tiket Masuk Weekend'], errors='coerce').fillna(0)
    df['Harga'] = ((df['Tiket Masuk Weekday'] + df['Tiket Masuk Weekend']) / 2).astype(int)

    # Deteksi kota semi otomatis
    daftar_kota = ['Jember', 'Malang', 'Batu']
    def deteksi_kota(lokasi):
        for kota in daftar_kota:
            if re.search(r'\b{}\b'.format(re.escape(kota)), lokasi, re.IGNORECASE):
                return kota
        return 'Tidak Diketahui'

    df['Kota'] = df['Lokasi'].apply(deteksi_kota)

    return df

df = load_data()

# -----------------------------
# FITUR: TF-IDF dan SIMILARITY
# -----------------------------
def get_similarity_matrix(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['Deskripsi'])
    return cosine_similarity(tfidf_matrix)

cosine_sim = get_similarity_matrix(df)

# -----------------------------
# FUNGSI REKOMENDASI
# -----------------------------
def recommend_places(kata_kunci, kota_filter=None, min_rating=0, min_harga=0, max_harga=float('inf'), top_n=5):
    # Cari baris paling mirip dengan kata kunci
    idx = df[df['Nama Wisata'].str.contains(kata_kunci, case=False, na=False)]
    if idx.empty:
        return pd.DataFrame(), []
    idx = idx.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]  # Skip diri sendiri

    recommended = []
    relevance_ground_truth = []

    for i in sim_scores:
        place = df.iloc[i[0]]

        # Filter kota yang lebih presisi (berbasis kolom 'Kota')
        if kota_filter:
            if kota_filter.lower() != place['Kota'].lower():
                continue

        # Filter rating dan harga
        if place['Rating'] < min_rating:
            continue
        if not (min_harga <= place['Harga'] <= max_harga):
            continue

        recommended.append({
            "Nama Wisata": place['Nama Wisata'],
            "Kota": place['Kota'],
            "Lokasi": place['Lokasi'],
            "Deskripsi": place['Deskripsi'],
            "Rating": place['Rating'],
            "Harga": place['Harga'],
            "Skor Kemiripan": round(i[1], 3)
        })
        relevance_ground_truth.append(1 if kata_kunci.lower() in place['Deskripsi'].lower() else 0)
        if len(recommended) == top_n:
            break

    return pd.DataFrame(recommended), relevance_ground_truth

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎯 Sistem Rekomendasi Tempat Wisata - Jawa Timur")

kata_kunci = st.text_input("Masukkan kata kunci/nama tempat wisata (misal: 'Papuma')", "")
kota_filter = st.selectbox("Filter kota (opsional)", [''] + sorted(df['Kota'].unique()))
min_rating = st.slider("Filter minimal rating", 0.0, 5.0, 0.0, 0.5)
min_harga = st.number_input("Filter harga minimal", 0, 1000000, 0, 1000)
max_harga = st.number_input("Filter harga maksimal", 0, 1000000, 1000000, 1000)
top_n = st.slider("Jumlah rekomendasi", 1, 10, 5)

if st.button("Cari Rekomendasi"):
    if not kata_kunci:
        st.warning("Silakan masukkan kata kunci terlebih dahulu.")
    else:
        hasil, ground_truth = recommend_places(kata_kunci, kota_filter if kota_filter else None, min_rating, min_harga, max_harga, top_n)
        if hasil.empty:
            st.error("Tempat tidak ditemukan. Coba kata kunci atau filter lain.")
        else:
            st.success(f"{len(hasil)} tempat wisata direkomendasikan.")
            st.dataframe(hasil)

            # Precision & Recall Evaluation (simulasi)
            relevan = sum(ground_truth)
            total = len(ground_truth)
            precision = relevan / total if total > 0 else 0
            recall = relevan / (relevan + 1e-5)  # Hindari div 0

            st.subheader("📊 Evaluasi Rekomendasi")
            st.write(f"**Precision**: {precision:.2f}")
            st.write(f"**Recall**: {recall:.2f}")
