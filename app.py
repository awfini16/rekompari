import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack

# -----------------------------
# LOAD DAN CLEAN DATASET
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("datasetku.csv", encoding='latin-1')
    df['Deskripsi'] = df['Deskripsi'].fillna('')
    df['Rating'] = df['Rating'].str.replace(',', '.').astype(float)
    df['Tiket Masuk Weekday'] = pd.to_numeric(df['Tiket Masuk Weekday'], errors='coerce').fillna(0)
    df['Tiket Masuk Weekend'] = pd.to_numeric(df['Tiket Masuk Weekend'], errors='coerce').fillna(0)
    df['Harga'] = ((df['Tiket Masuk Weekday'] + df['Tiket Masuk Weekend']) / 2).astype(int)
    return df

df = load_data()

# -----------------------------
# HYBRID SIMILARITY: TF-IDF + Rating + Harga
# -----------------------------
def get_similarity_matrix(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['Deskripsi'])

    scaler = MinMaxScaler()
    numeric_features = scaler.fit_transform(df[['Rating', 'Harga']])

    # Bobot fitur
    teks_weight = 0.5
    rating_weight = 0.3
    harga_weight = 0.2

    numeric_weighted = numeric_features * [rating_weight, harga_weight]
    combined_features = hstack([tfidf_matrix * teks_weight, numeric_weighted])

    return cosine_similarity(combined_features)

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
# METRIK EVALUASI
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


def plot_similarity(scores, labels):
    fig, ax = plt.subplots(figsize=(10, 1.5))
    sns.heatmap([scores], annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, xticklabels=labels, ax=ax)
    ax.set_title("Skor Kemiripan Tempat Rekomendasi", fontsize=14)
    ax.set_yticks([])
    plt.xticks(rotation=30, ha='right')
    return fig


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🗺️ Sistem Rekomendasi Tempat Wisata - Jawa Timur")

kata_kunci = st.text_input("Masukkan kata kunci/nama tempat wisata (misal: 'Papuma')", "")
min_rating = st.slider("Filter minimal rating", 0.0, 5.0, 0.0, 0.5)
min_harga = st.number_input("Filter harga minimal", 0, 1000000, 0, 1000)
max_harga = st.number_input("Filter harga maksimal", 0, 1000000, 1000000, 1000)
top_n = st.slider("Jumlah rekomendasi", 1, 10, 5)

# Visualisasi skor kemiripan
st.subheader("Visualisasi Skor Kemiripan")
fig = plot_similarity(hasil['Skor Kemiripan'].tolist(), hasil['Nama Wisata'].tolist())
st.pyplot(fig)
