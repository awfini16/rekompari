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
