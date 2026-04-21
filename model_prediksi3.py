import pandas as pd
import numpy as np
import logging
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# --- MEMBUAT FOLDER UNTUK GRAFIK SAMPEL ---
folder_grafik = 'grafik_prediksi'
if not os.path.exists(folder_grafik):
    os.makedirs(folder_grafik)

print("1. Membaca dataset asli...")
df = pd.read_excel('DATA2_GABUNGAN_7_POSYANDU.xlsx')

# Antisipasi jika ada spasi tersembunyi atau huruf kapital di nama kolom Excel
df.columns = df.columns.str.strip().str.lower()
df['tanggal_ukur'] = pd.to_datetime(df['tanggal_ukur'])

daftar_anak = df['nama_anak'].unique()
print(f"2. Dataset dibaca! Total ada {len(daftar_anak)} anak.")

hasil_evaluasi = {
    'berat': {'mae': [], 'rmse': [], 'mape': []},
    'tinggi': {'mae': [], 'rmse': [], 'mape': []},
    'lingkar_kepala': {'mae': [], 'rmse': [], 'mape': []}
}

detail_per_anak = []

# Batasan agar tidak semua anak dibuat grafiknya
maks_grafik = 5
grafik_tersimpan = 0

print("3. Mulai mengevaluasi seluruh anak per indikator. Mohon tunggu...")

for nama in daftar_anak:
    df_anak = df[df['nama_anak'] == nama].copy()
    df_anak = df_anak.dropna(subset=['tanggal_ukur']).sort_values('tanggal_ukur')
    
    if len(df_anak) < 5:
        continue 
        
    res_anak = {'Nama Anak': nama}
    
    for metrik in ['berat', 'tinggi', 'lingkar_kepala']:
        if metrik not in df_anak.columns:
            continue
            
        df_prophet = df_anak[['tanggal_ukur', metrik]].copy().rename(columns={'tanggal_ukur': 'ds', metrik: 'y'}).dropna()

        if len(df_prophet) < 5:
            continue

        train_size = int(len(df_prophet) * 0.8)
        data_latih = df_prophet.iloc[:train_size].copy()
        data_uji = df_prophet.iloc[train_size:].copy()
        
        if data_uji.empty:
            continue
        
        # Inisialisasi dan Training Model Prophet
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        model.fit(data_latih)
        
        # Prediksi
        forecast = model.predict(data_uji[['ds']])
        
        # Evaluasi Akurasi
        y_asli = data_uji['y'].values
        y_tebakan = forecast['yhat'].values
        
        error = y_asli - y_tebakan
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        mape = np.mean(np.abs(error / np.where(y_asli == 0, 1, y_asli))) * 100
        
        # Simpan ke rata-rata total
        hasil_evaluasi[metrik]['mae'].append(mae)
        hasil_evaluasi[metrik]['rmse'].append(rmse)
        hasil_evaluasi[metrik]['mape'].append(mape)
        
        # Simpan detail per anak untuk Excel
        res_anak[f'MAE {metrik}'] = mae
        res_anak[f'Akurasi {metrik} (%)'] = 100 - mape

        # --- SIMPAN GRAFIK SAMPEL UNTUK LAMPIRAN ---
        if grafik_tersimpan < maks_grafik:
            fig1 = model.plot(forecast)
            plt.title(f"Forecast {metrik.capitalize()} - {nama}")
            fig1.savefig(f'{folder_grafik}/{nama}_{metrik}_forecast.png')
            plt.close(fig1) 

            fig2 = model.plot_components(forecast)
            fig2.savefig(f'{folder_grafik}/{nama}_{metrik}_trend.png')
            plt.close(fig2)

    detail_per_anak.append(res_anak)
    
    if grafik_tersimpan < maks_grafik:
        grafik_tersimpan += 1

# --- SIMPAN HASIL DETAIL KE EXCEL ---
df_detail = pd.DataFrame(detail_per_anak)
df_detail.to_excel('LAPORAN_EVALUASI_PER_ANAK.xlsx', index=False)
print(f"\n[INFO] Detail evaluasi tiap anak disimpan di: LAPORAN_EVALUASI_PER_ANAK.xlsx")
print(f"[INFO] Grafik sampel disimpan di folder: {folder_grafik}/")

# --- OUTPUT TERMINAL SEPERTI SEBELUMNYA ---
print(f"\n" + "="*60)
print(f"{'HASIL EVALUASI AKHIR SISTEM PER INDIKATOR':^60}")
print("="*60)

indikator_list = []
akurasi_list = []

for m in hasil_evaluasi.keys():
    if hasil_evaluasi[m]['mae']:
        r_mae = np.mean(hasil_evaluasi[m]['mae'])
        r_rmse = np.mean(hasil_evaluasi[m]['rmse'])
        r_mape = np.mean(hasil_evaluasi[m]['mape'])
        akurasi = 100 - r_mape
        satuan = "kg" if m == 'berat' else "cm"
        
        # Simpan untuk grafik keseluruhan
        indikator_list.append(m.upper())
        akurasi_list.append(akurasi)
        
        print(f"--- INDIKATOR {m.upper()} ---")
        print(f"Rata-rata MAE  : {r_mae:.2f} {satuan}")
        print(f"Rata-rata RMSE : {r_rmse:.2f}")
        print(f"Rata-rata MAPE : {r_mape:.2f}%")
        print(f"⭐ AKURASI      : {akurasi:.2f}%")
        print("-" * 60)

# --- TAMBAHAN: GRAFIK KESELURUHAN SISTEM (BAR CHART) ---
plt.figure(figsize=(8, 5))
bars = plt.bar(indikator_list, akurasi_list, color=['#4CAF50', '#2196F3', '#FFC107'])
plt.title('Akurasi Keseluruhan Sistem Prediksi Pertumbuhan Balita', fontsize=14, fontweight='bold')
plt.ylabel('Akurasi (%)', fontsize=12)
plt.ylim(0, 105) # Batas atas sumbu Y agar teks persentase tidak terpotong

# Menambahkan angka persentase di atas setiap batang
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.savefig('GRAFIK_AKURASI_KESELURUHAN.png', bbox_inches='tight')
plt.close()
print(f"[INFO] Grafik Keseluruhan Sistem berhasil disimpan sebagai: GRAFIK_AKURASI_KESELURUHAN.png")