# Proposal Solusi

## Audio AI Forensics Analyzer

### Proposal Pengembangan dan Implementasi

**Tanggal:** 20 April 2026  
**Dokumen untuk:** Mitra eksternal / calon pengguna institusional / calon pendukung implementasi

---

## 1. Ringkasan Eksekutif

Audio AI Forensics Analyzer adalah dashboard analisis audio berbasis Streamlit yang dirancang untuk membantu proses **screening awal** terhadap file audio atau lagu, dengan tujuan menilai apakah karakteristik audio tersebut lebih dekat ke pola **Human**, **Hybrid**, atau **AI-generated**.

Solusi ini tidak diposisikan sebagai alat tuduhan otomatis, melainkan sebagai **screening assistant** yang menggabungkan:

- analisis fitur audio numerik
- logika forensik berbasis aturan konservatif
- model machine learning untuk kecenderungan klasifikasi
- evidence card yang memisahkan strong evidence, weak/context evidence, dan production-mimic indicators

Pendekatan ini membuat sistem lebih jujur, lebih transparan, dan lebih aman dipakai dibanding model “black box” yang hanya menampilkan label akhir tanpa penjelasan.

Saat ini, produk sudah berfungsi sebagai **local operational dashboard** untuk:

- analisis file audio
- penyimpanan hasil dan histori pemeriksaan
- penambahan data berlabel ke dataset
- sinkronisasi feature store
- retraining model dari dashboard

Dengan pengembangan lanjutan yang terarah, solusi ini berpotensi ditingkatkan menjadi **platform online** untuk kebutuhan demo, pilot project, maupun implementasi terkontrol.

---

## 2. Latar Belakang Masalah

Perkembangan generative audio dan AI music tools membuat proses verifikasi keaslian audio menjadi semakin menantang. Dalam banyak konteks, seperti kurasi konten, evaluasi submission lagu, audit materi kreatif, atau penyaringan internal, dibutuhkan alat bantu yang dapat:

- memberikan indikasi awal secara cepat
- tidak gegabah mengeluarkan vonis
- menunjukkan alasan teknis yang bisa ditinjau ulang
- mendukung workflow evaluasi manual

Kendala yang umum terjadi pada alat deteksi audio adalah:

- terlalu cepat menyimpulkan AI hanya dari karakter mixing modern
- angka probabilitas terlihat meyakinkan, tetapi tidak transparan
- tidak ada pemisahan antara sinyal kuat dan sinyal ambigu
- tidak ada fasilitas pembelajaran ulang dari data pengguna

Audio AI Forensics Analyzer dibangun untuk menjawab kebutuhan tersebut dengan prinsip utama:

**“lebih baik jujur dan transparan daripada terlihat pasti tetapi menyesatkan.”**

---

## 3. Tujuan Solusi

Tujuan utama pengembangan platform ini adalah:

- menyediakan dashboard screening audio yang praktis dan mudah dipakai
- membantu pengguna membaca hasil analisis dalam bahasa teknis maupun bahasa awam
- membangun dataset training bertahap dari hasil review manusia
- menjaga agar model dapat ditingkatkan tanpa wajib menyimpan semua audio mentah lama
- menyiapkan fondasi untuk migrasi dari local tool menjadi layanan online

---

## 4. Deskripsi Solusi

Audio AI Forensics Analyzer adalah aplikasi yang menganalisis file audio menggunakan kombinasi tiga lapis:

### 4.1. Lapisan Fitur Audio

Sistem mengekstrak fitur audio numerik seperti:

- spectral flatness
- spectral rolloff
- RMS energy
- zero crossing rate
- MFCC statistics
- phase coherence
- harmonic consistency
- long-range texture consistency
- high-frequency shimmer behavior
- quiet vs loud HF behavior
- fingerprint-style metrics untuk pola generator-like

### 4.2. Lapisan Forensik

Sistem kemudian membaca fitur tersebut dengan policy konservatif untuk menghasilkan:

- `PASS`
- `REVIEW`
- `FAIL`

Keputusan ini dipandu oleh evidence card yang memisahkan:

- `Strong evidence`
- `Weak/context evidence`
- `Production-mimic indicators`

Dengan cara ini, sistem tidak mudah menganggap mastering modern atau produksi yang sangat rapi sebagai bukti AI keras.

### 4.3. Lapisan Machine Learning

Model machine learning dipakai sebagai lapisan tambahan untuk membaca kecenderungan pola fitur secara global.

Saat ini model yang digunakan adalah:

- `LogisticRegression`
- dengan kalibrasi probabilitas
- serta feature store agar pengetahuan numerik tetap tersimpan walaupun audio lama sudah dibersihkan

Output ML dipisahkan dari keputusan forensik agar pengguna tidak salah mengartikan hasil.

---

## 5. Fitur Utama Produk Saat Ini

Berikut fitur yang sudah tersedia dalam versi saat ini:

- upload file audio untuk dianalisis
- tiga mode analisis:
  - Mode A: analisis lengkap + grafik
  - Mode B: analisis ringkas
  - Mode C: fokus humanization / arahan perbaikan
- hasil akhir yang dipisah menjadi:
  - forensic decision
  - prediksi ML
  - verdict probabilities
  - evidence card
  - fingerprint module
- penjelasan hasil dalam bahasa awam
- histori analisis yang bisa dibuka kembali
- tombol simpan file upload ke dataset berlabel:
  - Human
  - Hybrid
  - AI
- feature store untuk menyimpan fitur numerik training
- tombol sinkronisasi feature store
- retrain model dari dashboard
- penghapusan aman audio lama yang sudah tersimpan ke feature store
- launcher `.exe` agar aplikasi bisa dijalankan tanpa terminal

---

## 6. Nilai Tambah untuk Mitra Eksternal

Platform ini menawarkan beberapa nilai tambah yang relevan untuk calon mitra:

### 6.1. Transparansi

Sistem tidak hanya memberi label, tetapi juga menampilkan alasan teknis di balik hasil.

### 6.2. Operasional

Cocok untuk workflow screening awal sebelum review manual lebih lanjut.

### 6.3. Adaptif

Dataset dapat ditambah dan model dapat ditrain ulang sesuai kebutuhan domain pengguna.

### 6.4. Bertahap

Dapat dimulai sebagai local pilot tool, lalu ditingkatkan menjadi layanan online.

### 6.5. Human-in-the-loop

Sistem memberi ruang untuk koreksi manusia sehingga kualitas model dapat berkembang lebih sehat dari waktu ke waktu.

---

## 7. Kondisi Produk Saat Ini

Berdasarkan audit cepat terhadap implementasi saat ini, kondisi produk dapat dirangkum sebagai berikut:

### 7.1. Kesiapan Fungsional

- dashboard utama berjalan dengan baik
- import modul inti lolos
- launcher dan stop launcher tersedia
- training dari feature store tersedia
- histori analisis tersedia

### 7.2. Kesiapan Data

- feature store saat ini berisi **62 baris fitur numerik**
- audio mentah dataset saat ini tidak lagi menjadi sumber utama pembelajaran karena pengetahuan numerik sudah disimpan di feature store

### 7.3. Posisi Produk

Produk saat ini paling tepat diposisikan sebagai:

**“operational prototype / advanced local MVP”**

Artinya, produk sudah layak digunakan untuk workflow internal terbatas, tetapi masih memerlukan sejumlah penguatan sebelum diposisikan sebagai layanan online publik multi-user.

---

## 8. Keterbatasan yang Perlu Dijelaskan Secara Jujur

Untuk menjaga integritas produk saat diajukan ke pihak luar, berikut poin yang perlu disampaikan secara jujur:

- hasil analisis adalah **screening awal**, bukan alat kepastian hukum atau alat tuduhan otomatis
- akurasi model sangat dipengaruhi kualitas dan kebersihan dataset label
- lagu dengan mastering modern, edit vokal berat, atau karakter produksi tertentu dapat menyerupai pola AI
- audio panjang, instrumental, atau tanpa vokal dapat mengurangi stabilitas pembacaan pada metrik tertentu
- fingerprint generator-like bersifat probabilistik, bukan bukti absolut

Penyampaian keterbatasan ini justru memperkuat kredibilitas solusi di mata mitra yang serius.

---

## 9. Usulan Kolaborasi atau Implementasi

Solusi ini dapat diajukan ke pihak luar dalam beberapa bentuk kerja sama:

### Opsi A. Pilot Internal

Tujuan:

- dipakai oleh tim kecil
- menguji workflow screening
- mengumpulkan umpan balik dan sampel baru

Cocok untuk:

- label musik
- tim kurasi konten
- internal creative QA
- eksperimen riset

### Opsi B. Platform Semi-Produksi

Tujuan:

- dipakai oleh pengguna terbatas berbasis akun
- tersedia sebagai dashboard online dengan kontrol akses
- training dikelola oleh admin

### Opsi C. Produk Online Penuh

Tujuan:

- tersedia secara publik atau B2B
- mendukung manajemen user, penyimpanan audio, monitoring, audit trail, dan governance model

Tahap ini membutuhkan arsitektur yang lebih matang daripada implementasi lokal saat ini.

---

## 10. Masukan Jika Solusi Ini Ingin Dijadikan Online

Secara umum, produk ini **bisa dibawa online**, tetapi saya menyarankan pendekatan bertahap.

### 10.1. Apa yang Sudah Siap untuk Online

- UI berbasis Streamlit sudah cocok untuk demo atau pilot online
- workflow upload dan hasil analisis sudah cukup jelas
- feature store sudah menjadi fondasi penting untuk training yang tidak tergantung penuh pada audio lama

### 10.2. Apa yang Belum Ideal untuk Produksi Publik

Implementasi saat ini masih memiliki karakter local-app, misalnya:

- histori disimpan di file JSON lokal
- dataset index disimpan di file CSV lokal
- training dipicu langsung dari dashboard
- file audio dan feature store masih berbasis filesystem lokal
- belum ada autentikasi pengguna
- belum ada pemisahan role admin vs user umum
- belum ada object storage untuk upload online
- belum ada job queue untuk analisis berat dan training asynchronous

### 10.3. Rekomendasi Tahap Online

#### Tahap 1. Demo Online / Pilot Terbatas

Gunakan Streamlit sebagai front-end online dengan ruang lingkup:

- upload file
- analisis
- baca hasil
- histori terbatas

Sebaiknya:

- training dinonaktifkan untuk user umum
- sinkronisasi dataset dibatasi hanya untuk admin
- file upload tidak disimpan permanen kecuali diperlukan

Ini cocok untuk:

- presentasi
- pilot dengan mitra terbatas
- validasi minat pasar

#### Tahap 2. Semi-Produksi

Tambahkan:

- login / autentikasi
- penyimpanan metadata ke database
- object storage untuk file audio
- admin panel untuk retraining
- logging dan audit trail

#### Tahap 3. Produksi Penuh

Pisahkan arsitektur menjadi:

- front-end dashboard
- API/service layer
- background worker untuk ekstraksi fitur dan training
- database
- object storage
- monitoring dan security layer

Ini lebih tepat jika solusi ingin dijadikan layanan komersial atau dipakai lintas organisasi.

---

## 11. Rekomendasi Arsitektur Online

### Rekomendasi Paling Aman untuk Tahap Awal

**Pilot online terkontrol**

Struktur:

- Streamlit sebagai dashboard
- satu server aplikasi
- satu database ringan untuk metadata hasil
- object storage untuk file audio
- feature store tetap dipertahankan
- training hanya untuk admin

Keuntungannya:

- cepat diimplementasikan
- biaya awal lebih terkendali
- risiko operasional lebih rendah
- tetap cocok untuk presentasi ke pihak luar

### Rekomendasi untuk Jangka Menengah

**Pisahkan analisis dari training**

Struktur:

- user upload file
- file masuk storage
- worker menjalankan ekstraksi fitur
- hasil ditampilkan ke dashboard
- training dilakukan terjadwal atau manual oleh admin

Keuntungannya:

- lebih stabil
- lebih aman
- lebih siap jika jumlah pengguna bertambah

---

## 12. Risiko Implementasi Online

Jika dibawa online, beberapa risiko yang perlu diantisipasi:

- upload audio besar menyebabkan beban bandwidth dan storage
- analisis audio cukup berat di CPU
- training model dari dashboard publik berisiko mengganggu performa
- tanpa autentikasi dan hak akses, dataset bisa tercampur
- tanpa aturan retensi, file audio dapat membengkak di storage

Karena itu, online version sebaiknya dirancang dengan pembagian peran:

- user analisis
- reviewer
- admin dataset
- admin training

---

## 13. Roadmap Pengembangan yang Disarankan

### Fase 1. Presentasi dan Pilot

- stabilisasi dashboard
- perapihan UI dan branding
- pilot online terbatas
- validasi kebutuhan pengguna

### Fase 2. Data dan Governance

- perbaikan dataset label
- peningkatan kualitas feature store
- audit hasil dan evaluasi model
- pembatasan role pengguna

### Fase 3. Online Readiness

- database metadata
- object storage
- async processing
- monitoring performa
- secrets management

### Fase 4. Production Readiness

- auth
- audit trail
- deployment pipeline
- backup strategy
- SLA internal atau komersial

---

## 14. Penutup

Audio AI Forensics Analyzer memiliki potensi kuat untuk dikembangkan sebagai solusi screening audio yang transparan, adaptif, dan operasional.

Nilai utamanya bukan pada klaim “deteksi absolut”, melainkan pada kemampuannya untuk:

- memberi screening awal secara cepat
- memisahkan sinyal kuat dan sinyal ambigu
- menjelaskan hasil dengan bahasa yang bisa dipahami
- terus belajar dari dataset yang tumbuh bertahap

Dengan strategi implementasi yang bertahap, solusi ini layak diajukan ke pihak luar sebagai:

- pilot tool
- internal review platform
- atau fondasi produk online yang lebih matang di tahap berikutnya

---

## 15. Lampiran Singkat: Posisi Strategis Produk

### Cocok diajukan sebagai:

- alat bantu screening audio
- dashboard analisis forensik audio internal
- platform evaluasi awal untuk human / hybrid / AI audio

### Tidak disarankan diajukan sebagai:

- alat tuduhan otomatis
- alat kepastian hukum
- sistem final tanpa review manusia

### Narasi yang paling aman untuk pihak luar:

> Audio AI Forensics Analyzer adalah platform screening awal yang membantu membaca kecenderungan audio terhadap pola human, hybrid, atau AI-generated secara lebih transparan. Sistem ini menggabungkan analisis fitur audio, policy forensik konservatif, dan machine learning, serta tetap menempatkan review manusia sebagai bagian penting dalam pengambilan keputusan.

