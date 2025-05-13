import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# === 1. LOAD & PREPROCESS DATA ===
@st.cache_data
def load_data():
    df = pd.read_csv("datasetku.csv", encoding='latin-1')

    # Bersihkan dan ubah tipe data
    df.dropna(subset=['Deskripsi', 'Rating'], inplace=True)
    df['Rating'] = df['Rating'].str.replace(',', '.').astype(float)
    df['Tiket Masuk Weekday'] = df['Tiket Masuk Weekday'].replace("Gratis", 0).astype(int)
    df['Tiket Masuk Weekend'] = df['Tiket Masuk Weekend'].replace("Gratis", 0).astype(int)
    
    # Normalisasi fitur numerik
    scaler = MinMaxScaler()
    df[['Rating_norm', 'Tiket_Weekday_norm', 'Tiket_Weekend_norm']] = scaler.fit_transform(
        df[['Rating', 'Tiket Masuk Weekday', 'Tiket Masuk Weekend']]
    )
    
    return df

df = load_data()

# === 2. SIDEBAR - PREFERENSI PENGGUNA ===
st.sidebar.title("Preferensi Wisata")
kategori_input = st.sidebar.text_input("Kategori wisata (contoh: alam, budaya, kuliner)")
lokasi_input = st.sidebar.text_input("Lokasi yang diinginkan (contoh: Malang, Banyuwangi)")
fasilitas_input = st.sidebar.text_input("Fasilitas yang diinginkan (contoh: parkir, toilet)")
harga_max = st.sidebar.slider("Harga tiket maksimal (Weekday)", 0, 100000, 50000)
min_rating = st.sidebar.slider("Minimal rating", 0.0, 5.0, 3.5)

# Gabungkan preferensi menjadi satu string untuk TF-IDF
preferensi_text = f"{kategori_input} {lokasi_input} {fasilitas_input}"

# === 3. FITUR TF-IDF + NUMERIK ===
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['Deskripsi'].fillna(''))

# Vektor preferensi pengguna
user_vec = tfidf.transform([preferensi_text])

# Gabungkan dengan fitur numerik
user_profile = np.hstack([
    user_vec.toarray(),
    [[min_rating, harga_max/df['Tiket Masuk Weekday'].max(), harga_max/df['Tiket Masuk Weekend'].max()]]
])

feature_matrix = np.hstack([
    tfidf_matrix.toarray(),
    df[['Rating_norm', 'Tiket_Weekday_norm', 'Tiket_Weekend_norm']].values
])

# === 4. HITUNG SIMILARITY DAN REKOMENDASI ===
cos_sim = cosine_similarity(user_profile, feature_matrix)[0]
df['Similarity'] = cos_sim

# Filter dan sort
filtered = df[
    (df['Tiket Masuk Weekday'] <= harga_max) &
    (df['Rating'] >= min_rating)
].sort_values(by='Similarity', ascending=False).head(5)

# === 5. TAMPILKAN HASIL REKOMENDASI ===
st.title("🎯 Rekomendasi Tempat Wisata Jawa Timur")
st.markdown("Berikut adalah tempat wisata yang cocok dengan preferensimu:")

if not filtered.empty:
    for i, row in filtered.iterrows():
        st.subheader(row['Nama Wisata'])
        st.write(f"📍 Lokasi: {row['Lokasi']}")
        st.write(f"⭐ Rating: {row['Rating']} | 💰 Tiket: Rp{row['Tiket Masuk Weekday']} (weekday)")
        st.write(f"📝 {row['Deskripsi']}")
        st.markdown("---")
else:
    st.warning("Tidak ditemukan wisata yang sesuai dengan preferensimu.")
