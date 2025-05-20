            st.success(f"{len(hasil)} tempat wisata direkomendasikan.")
            st.dataframe(hasil)

            # Evaluasi rata-rata similarity score
            def average_similarity_score(df_hasil):
                if "Skor Kemiripan" in df_hasil.columns and not df_hasil.empty:
                    return round(df_hasil["Skor Kemiripan"].mean(), 3)
                return 0.0

            avg_score = average_similarity_score(hasil)
            st.success(f" Rata-rata Similarity Score: {avg_score}")

            # Visualisasi bar chart
            st.subheader(" Visualisasi Skor Kemiripan")
            st.bar_chart(hasil.set_index("Nama Wisata")["Skor Kemiripan"])

            # Precision & Recall sederhana
            relevan = sum(ground_truth)
            total = len(ground_truth)
            precision = relevan / total if total > 0 else 0
            recall = relevan / (relevan + 1e-5)

            st.subheader("📈 Evaluasi Rekomendasi")
            st.write(f"**Precision**: {precision:.2f}")
            st.write(f"**Recall**: {recall:.2f}")
