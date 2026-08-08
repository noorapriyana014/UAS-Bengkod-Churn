# Prediksi Churn Pelanggan (UAS Bengkel Koding) 📊📉

## Deskripsi Proyek
Proyek ini merupakan tugas Ujian Akhir Semester (UAS) untuk program Bengkel Koding. Repositori ini berisi eksplorasi data, pemrosesan, dan pembuatan model *Machine Learning* untuk memprediksi *Customer Churn* (probabilitas pelanggan meninggalkan layanan). 

Memprediksi *churn* sangat penting bagi bisnis untuk mempertahankan pelanggan dan mengambil tindakan pencegahan yang tepat sasaran sebelum pelanggan tersebut benar-benar pergi.

## Teknologi & Library
- **Bahasa Pemrograman:** Python
- **Environment:** Jupyter Notebook & Streamlit
- **Library Utama:** 
  - `pandas` & `numpy` (Manipulasi dan Analisis Data)
  - `matplotlib` & `seaborn` (Visualisasi Data)
  - `scikit-learn` (Pembuatan Model Machine Learning & Evaluasi)

## Alur Kerja (Workflow)
Di dalam `UAS_BENGKOD_final.ipynb`, proses penyelesaian masalah dibagi menjadi beberapa tahap:
1. **Data Preprocessing:** Membersihkan data, menangani nilai yang hilang (*missing values*), dan melakukan *encoding* pada data kategorikal.
2. **Exploratory Data Analysis (EDA):** Menganalisis pola dan memvisualisasikan data untuk memahami karakteristik pelanggan.
3. **Modeling:** Melatih data dan membandingkan dua algoritma Machine Learning, yaitu Logistic Regression dan Random Forest (keduanya telah melalui proses *Hyperparameter Tuning*).
4. **Evaluasi Model:** Mengukur performa menggunakan metrik evaluasi seperti Akurasi, Precision, Recall, dan F1-Score.

## Hasil dan Kesimpulan
Proyek ini membandingkan dua algoritma Machine Learning yang telah dioptimasi. Berdasarkan hasil pengujian:

- **Model Terbaik:** Random Forest
- **Akurasi Model:** **78%** (0.7801)
- Model Random Forest mengungguli Logistic Regression (akurasi 77.3%) dan dipilih sebagai model final (`single_best_model.pkl`) untuk aplikasi prediksi *churn*.

## Cara Menjalankan Proyek
Untuk menjalankan proyek ini secara lokal, ikuti langkah-langkah berikut:

1. Clone repositori ini: 
   `git clone https://github.com/noorapriyana014/UAS-Bengkod-Churn.git`
2. Install dependensi yang dibutuhkan:
   `pip install -r requirements.txt`
3. **Untuk melihat proses analisis (Notebook):** Jalankan perintah `jupyter notebook` dan buka file `UAS_BENGKOD_final.ipynb`.
4. **Untuk menjalankan antarmuka web (Aplikasi):** Jalankan perintah `streamlit run app.py` di terminal.
