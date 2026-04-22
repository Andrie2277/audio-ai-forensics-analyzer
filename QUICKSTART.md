# Quick Start

Panduan singkat untuk penggunaan harian.

## Menjalankan App

1. Double click `AudioAIForensicsLauncher.exe`
2. Tunggu browser terbuka ke `http://127.0.0.1:8501`

Untuk menutup server:
- double click `AudioAIForensicsStop.exe`

## Analisis Audio

1. Upload file audio
2. Pilih mode analisis
3. Klik `Jalankan Analisis`
4. Baca hasil utama:
   - `Forensic decision`
   - `Prediksi ML`
   - `Bahasa Sederhana`

## Menyimpan ke Dataset

Setelah analisis selesai:
- klik `Tambahkan ke Dataset Human`, atau
- klik `Tambahkan ke Dataset Hybrid`, atau
- klik `Tambahkan ke Dataset AI`

Ini akan:
- menyalin file ke folder dataset
- memperbarui `dataset.csv`
- menambahkan fitur ke `training_features.csv`

## Sinkronkan Feature Store

Jika kamu masih punya banyak file lama di folder `data`:

1. buka tab `Training & Dataset`
2. klik `Sinkronkan Feature Store dari Dataset Lama`
3. tunggu sampai selesai

## Training Ulang Model

1. buka tab `Training & Dataset`
2. klik `Train Model dari feature store`
3. tunggu sampai selesai

## Menghapus Audio Lama Dengan Aman

Di tab `Training & Dataset`:
- lihat jumlah file `Aman dihapus`
- centang konfirmasi
- klik `Hapus Audio Lama yang Sudah Aman`

## Jika Hapus Manual

Kalau kamu menghapus file langsung dari folder `data`:
- klik `Refresh Dataset Index`
- kalau perlu klik `Refresh Status File Skipped`

## Aturan Praktis

- jangan hapus file yang belum masuk feature store
- jangan masukkan file ke dataset kalau labelnya belum yakin
- file `skipped` biasanya perlu dikonversi ke `WAV`
