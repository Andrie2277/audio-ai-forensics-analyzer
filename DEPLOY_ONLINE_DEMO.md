# Deploy Online Demo

Panduan ini dipakai untuk menaikkan aplikasi ke mode online demo yang aman untuk publik.

## Tujuan Mode Demo

Mode demo publik hanya menampilkan fitur:
- upload audio
- pilih mode analisis
- jalankan analisis
- baca hasil screening
- baca halaman `How It Works`

Mode demo **menyembunyikan** fitur admin:
- training model
- simpan ke dataset
- sinkron feature store
- hapus audio lama
- history persisten di file lokal

## Langkah 1: Siapkan Repository

Pastikan file berikut sudah ada:
- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `model.joblib`

Kalau demo ingin memakai model yang sudah dilatih, file `model.joblib` harus ikut tersedia saat deploy.

## Langkah 2: Aktifkan Demo Mode

Set environment variable berikut saat deploy:

```text
AUDIO_ANALYZER_DEMO_MODE=1
```

Arti flag ini:
- aplikasi masuk ke mode publik
- tab `Training & Dataset` dan `History` disembunyikan
- tombol simpan ke dataset tidak ditampilkan

## Langkah 3: Deploy ke Streamlit Community Cloud

1. Push project ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih repository ini.
4. Set main file ke:

```text
app.py
```

5. Tambahkan secret / environment variable:

```text
AUDIO_ANALYZER_DEMO_MODE=1
```

6. Deploy aplikasi.

## Analytics Ringan (Opsional)

Mode online demo sekarang mendukung analytics ringan yang bisa diaktifkan lewat secrets.

### Opsi 1: Plausible (Disarankan)

Tambahkan secrets berikut:

```toml
AUDIO_ANALYZER_DEMO_MODE="1"
ANALYTICS_PROVIDER="plausible"
PLAUSIBLE_DOMAIN="audio-ai-forensics-analyzer.streamlit.app"
```

Kalau kamu memakai domain lain nanti, cukup ganti nilai `PLAUSIBLE_DOMAIN`.

### Opsi 2: Google Analytics 4

Tambahkan secrets berikut:

```toml
AUDIO_ANALYZER_DEMO_MODE="1"
ANALYTICS_PROVIDER="ga4"
GA_MEASUREMENT_ID="G-XXXXXXXXXX"
```

Catatan:
- analytics hanya aktif jika provider dan secret-nya diisi
- kalau secret analytics tidak diisi, app tetap berjalan normal
- mode demo tetap fokus ke analisis publik, bukan tracking identitas pengguna

### Event yang Otomatis Dicatat

Kalau analytics aktif, app juga akan mengirim event ringan berikut:

- `upload_audio`
- `change_mode`
- `run_analysis`

Tujuannya agar kamu tidak hanya melihat pageview, tetapi juga:
- berapa kali user upload file
- mode analisis mana yang paling sering dipakai
- berapa kali tombol analisis benar-benar dijalankan

## Catatan Penting: Python Version

Project ini memakai requirement Python:

```text
>=3.13
```

Jadi saat deploy di Streamlit Community Cloud:

1. klik `Advanced settings`
2. pilih Python `3.13`
3. baru lanjutkan deploy

Kalau dibiarkan default ke `3.12`, deploy bisa gagal karena versi Python tidak cocok.

## Langkah 4: Cek Setelah Online

Pastikan yang muncul:
- tab `Workspace`
- tab `How It Works`
- uploader audio
- tombol `Jalankan Analisis`

Pastikan yang **tidak** muncul:
- `Training & Dataset`
- `History`
- tombol `Tambahkan ke Dataset Human/Hybrid/AI`

## Rekomendasi Aman untuk Publik

- Gunakan versi ini sebagai **screening demo**, bukan alat vonis final.
- Jangan buka fitur training ke user publik.
- Jangan jadikan file upload publik sebagai tambahan dataset otomatis.
- Simpan teks peringatan bahwa hasil dibaca sebagai screening assistant.

## Checklist Singkat

- `model.joblib` tersedia
- `requirements.txt` tersedia
- `AUDIO_ANALYZER_DEMO_MODE=1`
- main file = `app.py`
- fitur admin tidak tampil
