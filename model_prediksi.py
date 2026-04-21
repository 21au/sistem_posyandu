import pandas as pd
from prophet import Prophet
import warnings

# Mengabaikan pesan warning dari Prophet agar terminal tetap rapi
warnings.filterwarnings('ignore')

print("1. Membaca dataset asli...")
# Ganti nama file ini sesuai dengan nama file CSV di komputermu
df = pd.read_excel('DATA_GABUNGAN_7_POSYANDU.xlsx')
df['tanggal_ukur'] = pd.to_datetime(df['tanggal_ukur'])

# Melihat daftar semua anak di dataset
daftar_anak = df['nama_anak'].unique()
print(f"2. Dataset berhasil dibaca! Terdapat total {len(daftar_anak)} anak yang berbeda di dalam data ini.")

# ==========================================
# FUNGSI DINAMIS UNTUK MEMPREDIKSI ANAK SIAPAPUN
# ==========================================
def prediksi_anak(dataset, nama_target, metrik='berat', bulan_kedepan=3):
    """
    Fungsi ini akan mencari data anak berdasarkan nama,
    lalu memprediksi pertumbuhannya.
    """
    # 1. Filter data hanya untuk anak yang dicari
    df_anak = dataset[dataset['nama_anak'] == nama_target].copy()
    
    # Cek apakah anak tersebut ada di data dan datanya cukup
    if len(df_anak) < 2:
        return f"Maaf, data untuk {nama_target} kurang dari 2 bulan, tidak bisa diprediksi."
        
    df_anak = df_anak.sort_values('tanggal_ukur')
    
    # 2. Siapkan data untuk Prophet
    df_prophet = df_anak[['tanggal_ukur', metrik]].copy()
    df_prophet.rename(columns={'tanggal_ukur': 'ds', metrik: 'y'}, inplace=True)
    
    # 3. Buat dan Latih Model
    model = Prophet(interval_width=0.95, yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_prophet)
    
    # 4. Prediksi
    future = model.make_future_dataframe(periods=bulan_kedepan, freq='MS')
    forecast = model.predict(future)
    
    # 5. Ambil hasil prediksi masa depan
    hasil = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(bulan_kedepan)
    
    # Rapikan hasil (bulatkan 2 angka di belakang koma agar rapi)
    hasil['yhat'] = hasil['yhat'].round(2)
    hasil['yhat_lower'] = hasil['yhat_lower'].round(2)
    hasil['yhat_upper'] = hasil['yhat_upper'].round(2)
    
    return hasil

# ==========================================
# SIMULASI PENGGUNAAN DI SISTEM
# ==========================================
print("\n3. Mari kita coba prediksi untuk beberapa anak berbeda:")

# Kita coba panggil fungsi untuk anak pertama
anak_1 = 'A. FIRDAUSY NUZULA'
print(f"\n--- PREDIKSI BERAT BADAN (KG): {anak_1} ---")
hasil_1 = prediksi_anak(df, nama_target=anak_1, metrik='berat', bulan_kedepan=3)
print(hasil_1)

# Kita coba panggil fungsi untuk anak yang lain yang ada di datamu
anak_2 = 'UMAR ABDILLAH'
print(f"\n--- PREDIKSI BERAT BADAN (KG): {anak_2} ---")
hasil_2 = prediksi_anak(df, nama_target=anak_2, metrik='berat', bulan_kedepan=3)
print(hasil_2)

# Kalau mau prediksi Tinggi Badan Umar Abdillah, tinggal ganti metriknya:
print(f"\n--- PREDIKSI TINGGI BADAN (CM): {anak_2} ---")
hasil_3 = prediksi_anak(df, nama_target=anak_2, metrik='tinggi', bulan_kedepan=3)
print(hasil_3)