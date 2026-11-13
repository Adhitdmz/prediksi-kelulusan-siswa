import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

FILE_MODEL = BASE_DIR / "model_prediksi_siswa.pkl"
FILE_RIWAYAT = BASE_DIR / "riwayat_prediksi.csv"

st.set_page_config(
    page_title="Prediksi Kelulusan Siswa",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# MEMUAT MODEL MACHINE LEARNING
# ==========================================

model = joblib.load(FILE_MODEL)


# ==========================================
# FUNGSI PENYIMPANAN RIWAYAT
# ==========================================

def simpan_riwayat(data_baru):

    if FILE_RIWAYAT.exists():

        riwayat = pd.read_csv(FILE_RIWAYAT)

        riwayat = pd.concat(
            [riwayat, pd.DataFrame([data_baru])],
            ignore_index=True
        )

    else:

        riwayat = pd.DataFrame([data_baru])

    riwayat.to_csv(FILE_RIWAYAT, index=False)


# ==========================================
# CSS DESAIN APLIKASI
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.header {
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    padding: 35px;
    border-radius: 15px;
    color: white;
    margin-bottom: 25px;
}

.header h1 {
    color: white;
    font-size: 35px;
}

.header p {
    color: #dbeafe;
    font-size: 16px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}

.result-success {
    background-color: #dcfce7;
    color: #166534;
    padding: 25px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
}

.result-failed {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 25px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
st.markdown("""
<div class="header">
    <h1>🎓 Prediksi Kelulusan Siswa</h1>
    <p>
        Sistem Machine Learning untuk membantu memprediksi
        hasil akademik siswa menggunakan algoritma Decision Tree.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# DASHBOARD INFORMASI
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algoritma", "Decision Tree")

with col2:
    st.metric("Jenis Sistem", "Klasifikasi")

with col3:
    st.metric("Status", "Aktif 🟢")


st.divider()


# ==========================================
# FORM INPUT SISWA
# ==========================================

st.subheader("📋 Informasi Siswa")

st.write(
    "Masukkan informasi akademik siswa "
    "untuk melakukan prediksi kelulusan."
)

with st.form("form_prediksi"):

    nama = st.text_input("Nama Siswa")

    col1, col2 = st.columns(2)

    with col1:

        waktu_belajar = st.number_input(
            "Waktu Belajar",
            min_value=1,
            max_value=4,
            value=2,
            help="Kategori waktu belajar dari 1 sampai 4."
        )

        ketidakhadiran = st.number_input(
            "Jumlah Ketidakhadiran",
            min_value=0,
            max_value=100,
            value=0
        )

    with col2:

        g1 = st.number_input(
            "Nilai G1",
            min_value=0,
            max_value=20,
            value=10
        )

        g2 = st.number_input(
            "Nilai G2",
            min_value=0,
            max_value=20,
            value=10
        )

    tombol_prediksi = st.form_submit_button(
        "🔍 Prediksi Kelulusan",
        use_container_width=True
    )


# ==========================================
# PROSES PREDIKSI
# ==========================================

if tombol_prediksi:

    if nama.strip() == "":

        st.warning(
            "Silakan masukkan nama siswa terlebih dahulu."
        )

    else:

        data = pd.DataFrame(
            [[
                waktu_belajar,
                ketidakhadiran,
                g1,
                g2
            ]],
            columns=[
                "studytime",
                "absences",
                "G1",
                "G2"
            ]
        )

        hasil = model.predict(data)[0]

        # MENYIMPAN RIWAYAT

        data_riwayat = {

            "Tanggal": datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),

            "Nama Siswa": nama,

            "Waktu Belajar": waktu_belajar,

            "Ketidakhadiran": ketidakhadiran,

            "Nilai G1": g1,

            "Nilai G2": g2,

            "Hasil Prediksi": (
                "Lulus" if hasil == 1 else "Tidak Lulus"
            )
        }

        simpan_riwayat(data_riwayat)

        # MENAMPILKAN HASIL

        st.divider()

        st.subheader("📊 Hasil Prediksi")

        st.write(f"Nama Siswa: **{nama}**")

        if hasil == 1:

            st.markdown("""
            <div class="result-success">
                🎉 Siswa Memenuhi Batas Nilai Kelulusan
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-failed">
                ⚠️ Siswa Belum Memenuhi Batas Nilai Kelulusan
            </div>
            """, unsafe_allow_html=True)

        # RINGKASAN DATA

        st.subheader("📌 Ringkasan Data Siswa")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Waktu Belajar", waktu_belajar)

        col2.metric("Ketidakhadiran", ketidakhadiran)

        col3.metric("Nilai G1", g1)

        col4.metric("Nilai G2", g2)

        st.success("Riwayat prediksi berhasil disimpan!")


# ==========================================
# RIWAYAT PREDIKSI SISWA
# ==========================================

st.divider()

st.subheader("📚 Riwayat Prediksi Siswa")

if FILE_RIWAYAT.exists():

    riwayat = pd.read_csv(FILE_RIWAYAT)

    if not riwayat.empty:

        st.write(
            "Daftar seluruh siswa yang telah melakukan prediksi."
        )

        # STATISTIK RIWAYAT

        total_siswa = len(riwayat)

        total_lulus = (
            riwayat["Hasil Prediksi"] == "Lulus"
        ).sum()

        total_tidak_lulus = (
            riwayat["Hasil Prediksi"] == "Tidak Lulus"
        ).sum()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Prediksi",
            total_siswa
        )

        col2.metric(
            "Siswa Lulus",
            int(total_lulus)
        )

        col3.metric(
            "Siswa Tidak Lulus",
            int(total_tidak_lulus)
        )

        st.divider()

        # PENCARIAN SISWA

        pencarian = st.text_input(
            "🔎 Cari Nama Siswa",
            placeholder="Masukkan nama siswa..."
        )

        if pencarian:

            data_tampil = riwayat[
                riwayat["Nama Siswa"].str.contains(
                    pencarian,
                    case=False,
                    na=False
                )
            ]

        else:

            data_tampil = riwayat

        # TABEL RIWAYAT

        st.dataframe(
            data_tampil,
            use_container_width=True,
            hide_index=True
        )

        # GRAFIK PREDIKSI

        st.subheader("📈 Grafik Hasil Prediksi")

        grafik = pd.DataFrame({
            "Kategori": [
                "Lulus",
                "Tidak Lulus"
            ],

            "Jumlah": [
                int(total_lulus),
                int(total_tidak_lulus)
            ]
        })

        st.bar_chart(
            grafik.set_index("Kategori")
        )

       # ==========================================
# DOWNLOAD LAPORAN EXCEL
# ==========================================

st.subheader("📥 Download Laporan")

from io import BytesIO

output = BytesIO()

with pd.ExcelWriter(output, engine="openpyxl") as writer:
    riwayat.to_excel(
        writer,
        index=False,
        sheet_name="Riwayat Prediksi"
    )

st.download_button(
    label="📊 Download Laporan Excel",
    data=output.getvalue(),
    file_name="Laporan_Prediksi_Siswa.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "© 2026 | Aplikasi Prediksi Kelulusan Siswa "
    "| Machine Learning - Decision Tree"
)
