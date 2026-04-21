import pandas as pd
import numpy as np
import logging
from prophet import Prophet
import warnings

# Mengabaikan pesan error agar terminal bersih
warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

print("1. Membaca dataset asli...")
df = pd.read_excel('DATA2_GABUNGAN_7_POSYANDU.xlsx')
df['tanggal_ukur'] = pd.to_datetime(df['tanggal_ukur'])

daftar_anak = df['nama_anak'].unique()
print(f"2. Dataset dibaca! Total ada {len(daftar_anak)} anak.")

# ==========================================
# PERSIAPAN WADAH EVALUASI PER INDIKATOR
# ==========================================
# Menggunakan dictionary agar data tersimpan rapi berdasarkan kategorinya
hasil_evaluasi = {
    'berat': {'mae': [], 'rmse': [], 'mape': []},
    'tinggi': {'mae': [], 'rmse': [], 'mape': []},
    'lingkar_kepala': {'mae': [], 'rmse': [], 'mape': []}
}

print("3. Mulai mengevaluasi seluruh anak per indikator. Mohon tunggu...")

# Looping untuk setiap anak
for nama in daftar_anak:
    df_anak = df[df['nama_anak'] == nama].copy()
    df_anak = df_anak.dropna(subset=['tanggal_ukur']).sort_values('tanggal_ukur')
    
    # Minimal butuh 5 data untuk validasi split 80:20
    if len(df_anak) < 5:
        continue 
        
    metrik_pengukuran = ['berat', 'tinggi', 'lingkar_kepala']

    for metrik in metrik_pengukuran:
        if metrik not in df_anak.columns:
            continue
            
        # Siapkan dataframe Prophet
        df_prophet = df_anak[['tanggal_ukur', metrik]].copy()
        df_prophet.rename(columns={'tanggal_ukur': 'ds', metrik: 'y'}, inplace=True)
        df_prophet = df_prophet.dropna()

        # Pastikan data cukup setelah dropna
        if len(df_prophet) < 5:
            continue

        train_size = int(len(df_prophet) * 0.8)
        data_latih = df_prophet.iloc[:train_size].copy()
        data_uji = df_prophet.iloc[train_size:].copy()
        
        if data_uji.empty:
            continue
        
        # Latih model Prophet
        model = Prophet(yearly_seasonality=False, 
                        weekly_seasonality=False, 
                        daily_seasonality=False)
        model.fit(data_latih)
        
        # Prediksi
        future = data_uji[['ds']].reset_index(drop=True)
        forecast = model.predict(future)
        
        y_asli = data_uji['y'].values
        y_tebakan = forecast['yhat'].values
        
        # Hitung error
        error = y_asli - y_tebakan
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        mape = np.mean(np.abs(error / np.where(y_asli == 0, 1, y_asli))) * 100
        
        # Masukkan ke wadah sesuai metriknya
        hasil_evaluasi[metrik]['mae'].append(mae)
        hasil_evaluasi[metrik]['rmse'].append(rmse)
        hasil_evaluasi[metrik]['mape'].append(mape)

# ==========================================
# TAMPILKAN HASIL AKHIR PER INDIKATOR
# ==========================================
print(f"\n" + "="*60)
print(f"{'HASIL EVALUASI AKHIR SISTEM PER INDIKATOR':^60}")
print("="*60)

for m in hasil_evaluasi.keys():
    if hasil_evaluasi[m]['mae']: # Cek jika ada data yang berhasil dihitung
        r_mae = np.mean(hasil_evaluasi[m]['mae'])
        r_rmse = np.mean(hasil_evaluasi[m]['rmse'])
        r_mape = np.mean(hasil_evaluasi[m]['mape'])
        akurasi = 100 - r_mape
        satuan = "kg" if m == 'berat' else "cm"
        
        print(f"--- INDIKATOR {m.upper()} ---")
        print(f"Rata-rata MAE  : {r_mae:.2f} {satuan}")
        print(f"Rata-rata RMSE : {r_rmse:.2f}")
        print(f"Rata-rata MAPE : {r_mape:.2f}%")
        print(f"⭐ AKURASI      : {akurasi:.2f}%")
        print("-" * 60)

print(f"{'Selesai!':^60}")
print("="*60)