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
def recommend_places(kata_kunci, min_rating=0, min_harga=0, max_harga=float('inf'), top_n=5):
    idx = df[df['Nama Wisata'].str.contains(kata_kunci, case=False, na=False)]
    if idx.empty:
        return pd.DataFrame(), []
    idx = idx.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]

    recommended = []
    relevance_ground_truth = []

    for i in sim_scores:
        place = df.iloc[i[0]]

        if place['Rating'] < min_rating:
            continue
        if not (min_harga <= place['Harga'] <= max_harga):
            continue

        recommended.append({
            "Nama Wisata": place['Nama Wisata'],
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
# EVALUASI LANJUTAN
# -----------------------------
def precision_at_k(relevance, k):
    relevance = np.array(relevance)[:k]
    if relevance.size == 0:
        return 0.0
    return np.sum(relevance) / k

def average_precision(relevance):
    relevance = np.array(relevance)
    if np.sum(relevance) == 0:
        return 0.0
    score = 0.0
    hit_count = 0
    for i, rel in enumerate(relevance):
        if rel:
            hit_count += 1
            score += hit_count / (i + 1)
    return score / np.sum(relevance)

def dcg_at_k(relevance, k):
    relevance = np.array(relevance)[:k]
    if relevance.size == 0:
        return 0.0
    return np.sum((2 ** relevance - 1) / np.log2(np.arange(2, relevance.size + 2)))

def ndcg_at_k(relevance, k):
    actual_dcg = dcg_at_k(relevance, k)
    ideal_relevance = sorted(relevance, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevance, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

def evaluate_recommendation(relevance_list, k=5):
    p_at_k = precision_at_k(relevance_list, k)
    ap = average_precision(relevance_list)
    ndcg = ndcg_at_k(relevance_list, k)

    return {
        f'Precision@{k}': round(p_at_k, 3),
        'MAP': round(ap, 3),
        f'NDCG@{k}': round(ndcg, 3)
    }

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title(" Sistem Rekomendasi Tempat Wisata - Jawa Timur")

kata_kunci = st.text_input("Masukkan kata kunci/nama tempat wisata (misal: 'Papuma')", "")
min_rating = st.slider("Filter minimal rating", 0.0, 5.0, 0.0, 0.5)
min_harga = st.number_input("Filter harga minimal", 0, 1000000, 0, 1000)
max_harga = st.number_input("Filter harga maksimal", 0, 1000000, 1000000, 1000)
top_n = st.slider("Jumlah rekomendasi", 1, 10, 5)

if st.button("Cari Rekomendasi"):
    if not kata_kunci:
        st.warning("Silakan masukkan kata kunci terlebih dahulu.")
    else:
        hasil, ground_truth = recommend_places(kata_kunci, min_rating, min_harga, max_harga, top_n)
        if hasil.empty:
            st.error("Tempat tidak ditemukan. Coba kata kunci atau filter lain.")
        else:
            st.success(f"{len(hasil)} tempat wisata direkomendasikan.")
            st.dataframe(hasil)

            # Fungsi rata-rata similarity score
            def average_similarity_score(df_hasil):
                if "Skor Kemiripan" in df_hasil.columns and not df_hasil.empty:
                    return round(df_hasil["Skor Kemiripan"].mean(), 3)
                return 0.0

            avg_score = average_similarity_score(hasil)
            st.info(f"🔍 Rata-rata Skor Kemiripan: {avg_score}")

            # Visualisasi bar chart skor kemiripan
            st.subheader("📊 Visualisasi Skor Kemiripan")
            st.bar_chart(hasil.set_index("Nama Wisata")["Skor Kemiripan"])

            # Precision & Recall sederhana
            relevan = sum(ground_truth)
            total = len(ground_truth)
            precision = relevan / total if total > 0 else 0
            recall = relevan / (relevan + 1e-5)

            st.subheader("📈 Evaluasi Rekomendasi")
            st.write(f"**Precision**: {precision:.2f}")
            st.write(f"**Recall**: {recall:.2f}")

            # Evaluasi lanjutan (MAP, NDCG, dll)
            eval_result = evaluate_recommendation(ground_truth, top_n)
            for metric, score in eval_result.items():
                st.write(f"**{metric}**: {score}")
