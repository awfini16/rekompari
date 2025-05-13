import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("datasetku.csv", encoding='latin-1')
    df['Deskripsi'] = df['Deskripsi'].fillna('')
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
def recommend_places(kata_kunci, lokasi_filter=None, top_n=5):
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
        if lokasi_filter:
            if lokasi_filter.lower() not in place['Lokasi'].lower():
                continue
        recommended.append({
            "Nama Wisata": place['Nama Wisata'],
            "Lokasi": place['Lokasi'],
            "Deskripsi": place['Deskripsi'],
            "Rating": place['Rating'],
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
lokasi_filter = st.text_input("Filter lokasi (opsional, misal: 'Jember')", "")
top_n = st.slider("Jumlah rekomendasi", 1, 10, 5)

if st.button("Cari Rekomendasi"):
    if not kata_kunci:
        st.warning("Silakan masukkan kata kunci terlebih dahulu.")
    else:
        hasil, ground_truth = recommend_places(kata_kunci, lokasi_filter, top_n)
        if hasil.empty:
            st.error("Tempat tidak ditemukan. Coba kata kunci lain.")
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
