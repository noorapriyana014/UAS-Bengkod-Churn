import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. SETUP & LOAD MODEL ---
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📱", layout="wide")

@st.cache_resource
def load_model():
    try:
        model_dict = joblib.load('single_best_model.pkl')
        return model_dict
    except Exception as e:
        st.error(f"Error Loading Model: {e}")
        return None

data_model = load_model()

if data_model is not None:
    model = data_model['model']
    scaler = data_model['scaler']
    feature_names = data_model['columns']
else:
    st.stop()

# --- 2. SIDEBAR: INPUT USER ---
st.sidebar.header("📝 Masukkan Data Pelanggan")
st.sidebar.markdown("Sesuaikan nilai fitur di bawah ini:")

with st.sidebar.form("churn_form"):
    st.subheader("Profil Demografis")
    
    # --- PERUBAHAN DISINI: Gender ditaruh paling atas ---
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    partner = st.selectbox("Partner?", ["No", "Yes"])
    dependents = st.selectbox("Dependents?", ["No", "Yes"])

    st.subheader("Layanan Berlangganan")
    tenure = st.number_input("Lama Langganan (Bulan)", 0, 100, 12)
    phone_service = st.selectbox("Phone Service?", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines?", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service?", ["DSL", "Fiber optic", "No"])

    st.subheader("Keamanan & Support")
    online_security = st.selectbox("Online Security?", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup?", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection?", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support?", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies?", ["No", "Yes", "No internet service"])

    st.subheader("Akun & Tagihan")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing?", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Biaya Bulanan", 0.0, value=50.0)
    
    # Estimasi Total Charges otomatis jika user tidak mengubahnya
    total_charges = st.number_input("Total Biaya (Total Charges)", 0.0, value=monthly_charges * tenure)

    submitted = st.form_submit_button("🔍 Prediksi Sekarang")

# --- 3. MAIN PAGE ---
st.title("📱 Dashboard Prediksi Churn Pelanggan")
st.markdown("""
Aplikasi ini memprediksi kemungkinan pelanggan berhenti berlangganan (Churn) berdasarkan data profil dan pola penggunaan.
""")

# Membuat Tab untuk kerapian
tab1, tab2, tab3 = st.tabs(["📊 Hasil Prediksi", "📈 Insight Model", "ℹ️ Penjelasan Fitur"])

# --- TAB 1: PREDIKSI ---
with tab1:
    if submitted:
        # 1. Buat DataFrame Kosong dengan kolom yang sesuai model
        input_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

        # 2. Isi Data Numerik
        # Pastikan nama kolom numerik sesuai dengan yang ada di feature_names
        if 'tenure' in input_data.columns: input_data['tenure'] = tenure
        if 'MonthlyCharges' in input_data.columns: input_data['MonthlyCharges'] = monthly_charges
        if 'TotalCharges' in input_data.columns: input_data['TotalCharges'] = total_charges
        
        # Jika menggunakan input SeniorCitizen sebagai numerik/biner langsung
        if 'SeniorCitizen' in input_data.columns: input_data['SeniorCitizen'] = senior_citizen

        # 3. Handling One-Hot Encoding (Manual Mapping)
        # Fungsi untuk set nilai 1 pada kolom one-hot yang sesuai
        def set_ohe(raw_val, prefix):
            # Coba format "Prefix_Value" (Standard pandas get_dummies)
            col_name = f"{prefix}_{raw_val}"
            if col_name in input_data.columns:
                input_data[col_name] = 1
            # Coba format tanpa underscore jika ada variasi
            elif f"{prefix}{raw_val}" in input_data.columns:
                 input_data[f"{prefix}{raw_val}"] = 1
        
        # Terapkan mapping
        set_ohe(gender, "gender")
        set_ohe(partner, "Partner")
        set_ohe(dependents, "Dependents")
        set_ohe(phone_service, "PhoneService")
        set_ohe(multiple_lines, "MultipleLines")
        set_ohe(internet_service, "InternetService")
        set_ohe(online_security, "OnlineSecurity")
        set_ohe(online_backup, "OnlineBackup")
        set_ohe(device_protection, "DeviceProtection")
        set_ohe(tech_support, "TechSupport")
        set_ohe(streaming_tv, "StreamingTV")
        set_ohe(streaming_movies, "StreamingMovies")
        set_ohe(contract, "Contract")
        set_ohe(paperless_billing, "PaperlessBilling")
        set_ohe(payment_method, "PaymentMethod")

        # 4. Scaling
        # Transformasi seluruh baris data menggunakan scaler yang disimpan
        try:
            input_scaled = scaler.transform(input_data)
        except Exception as e:
            st.warning("Terjadi masalah saat scaling data. Mencoba prediksi tanpa scaling...")
            input_scaled = input_data

        # 5. Prediksi
        try:
            prediction = model.predict(input_scaled)[0]
            # Coba dapatkan probabilitas jika model mendukung
            if hasattr(model, "predict_proba"):
                prediction_proba = model.predict_proba(input_scaled)[0][1]
            else:
                prediction_proba = prediction # Fallback jika tidak ada proba
            
            # Tampilan Hasil
            st.divider()
            col_hasil, col_gauge = st.columns([2, 1])

            with col_hasil:
                st.subheader("Keputusan Model")
                if prediction == 1:
                    st.error(f"⚠️ **PREDIKSI: CHURN**")
                    st.write("Pelanggan ini memiliki risiko tinggi untuk berhenti berlangganan.")
                    st.warning("Saran: Tawarkan insentif atau hubungi pelanggan ini segera.")
                else:
                    st.success(f"✅ **PREDIKSI: TIDAK CHURN**")
                    st.write("Pelanggan ini diprediksi akan tetap setia.")
                    st.info("Saran: Pertahankan kualitas layanan.")

            with col_gauge:
                st.metric(label="Probabilitas Risiko", value=f"{prediction_proba*100:.2f}%")
                st.progress(prediction_proba)
                if prediction_proba > 0.5:
                    st.caption("🔴 Risiko Tinggi")
                else:
                    st.caption("🟢 Risiko Rendah")
        
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")

    else:
        st.info("👈 Silakan isi data pelanggan di Sidebar lalu klik 'Prediksi Sekarang'")

# --- TAB 2: INSIGHT MODEL ---
with tab2:
    st.header("Faktor Pengaruh (Feature Importance)")
    st.write("Grafik ini menunjukkan fitur mana yang paling dianggap penting oleh model.")

    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        df_imp = pd.DataFrame({'Fitur': feature_names, 'Pentingnya': imp}).sort_values('Pentingnya', ascending=False)
        st.bar_chart(df_imp.set_index('Fitur').head(10))
        st.caption("Fitur dengan bar terpanjang adalah yang paling mempengaruhi keputusan model.")
        
    elif hasattr(model, 'coef_'):
        imp = model.coef_[0]
        df_imp = pd.DataFrame({'Fitur': feature_names, 'Bobot': imp}).sort_values('Bobot', ascending=False)
        st.bar_chart(df_imp.set_index('Fitur').head(10))
        st.caption("Nilai positif meningkatkan risiko Churn, negatif menurunkan risiko.")
    
    else:
        st.warning("Model yang digunakan tidak mendukung visualisasi Feature Importance secara langsung.")

# --- TAB 3: PENJELASAN FITUR ---
with tab3:
    st.header("Kamus Data")
    with st.expander("Lihat Penjelasan Detail Fitur"):
        st.markdown("""
        * **Tenure:** Berapa bulan pelanggan telah berlangganan.
        * **MonthlyCharges:** Jumlah tagihan yang harus dibayar setiap bulan.
        * **TotalCharges:** Total biaya yang sudah dibayarkan selama menjadi pelanggan.
        * **Contract:** Jenis perjanjian berlangganan (Bulanan/Tahunan).
        * **InternetService:** Jenis layanan internet.
        * **PaymentMethod:** Cara pelanggan membayar tagihan.
        * **OnlineSecurity, TechSupport, dll:** Layanan tambahan yang diambil pelanggan.
        """)