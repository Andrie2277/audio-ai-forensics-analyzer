# Buku Panduan Belajar

## Audio AI Forensics Analyzer untuk Siswa SMA

**Versi sederhana untuk belajar, mencoba, dan memahami hasil analisis audio**

---

## Daftar Isi

1. Apa itu aplikasi ini?
2. Kenapa aplikasi ini dibuat?
3. Hal penting yang harus dipahami dari awal
4. Mengenal istilah dasar
5. Cara menjalankan aplikasi
6. Cara memakai aplikasi langkah demi langkah
7. Cara membaca hasil analisis
8. Arti istilah Human, Hybrid, dan AI
9. Apa itu forensic decision?
10. Apa itu fingerprint module?
11. Contoh membaca hasil dengan bahasa sederhana
12. Latihan belajar untuk siswa
13. Cara menambah data dan melatih model
14. Hal yang boleh dan tidak boleh dilakukan
15. Pertanyaan yang sering muncul
16. Penutup

---

## 1. Apa itu aplikasi ini?

Audio AI Forensics Analyzer adalah aplikasi untuk membantu membaca apakah sebuah audio atau lagu:

- lebih mirip dibuat manusia (`Human`)
- campuran manusia dan AI (`Hybrid`)
- atau lebih dekat ke pola audio hasil AI (`AI`)

Aplikasi ini **bukan alat sulap** dan **bukan alat penentu mutlak**.  
Fungsinya adalah sebagai **alat bantu screening awal**.

Artinya, aplikasi ini membantu kita:

- melihat tanda-tanda tertentu pada audio
- membaca kecenderungan hasil
- memahami apakah audio perlu dicek lagi secara manual

---

## 2. Kenapa aplikasi ini dibuat?

Sekarang banyak tools AI yang bisa membuat:

- lagu
- suara vokal
- backing music
- instrumental
- efek audio

Karena itu, kadang sulit membedakan:

- mana audio yang benar-benar dibuat manusia
- mana yang sudah dibantu AI
- mana yang terdengar sangat rapi karena editing modern

Aplikasi ini dibuat untuk membantu proses belajar dan pemeriksaan awal, terutama agar pengguna bisa lebih kritis saat mendengar dan menilai audio.

---

## 3. Hal penting yang harus dipahami dari awal

Sebelum memakai aplikasi ini, pahami 5 hal berikut:

1. Aplikasi ini tidak selalu benar 100%.
2. Hasil analisis harus dibaca dengan tenang, bukan langsung dipercaya mentah-mentah.
3. Audio yang sangat rapi belum tentu AI.
4. Lagu manusia yang dimixing sangat modern bisa terlihat "mirip AI".
5. Keputusan akhir tetap perlu logika manusia.

Kalimat paling penting:

> Aplikasi ini adalah **asisten screening**, bukan hakim terakhir.

---

## 4. Mengenal istilah dasar

Berikut istilah yang paling sering muncul:

### Audio
Suara yang direkam atau dibuat, misalnya lagu `.mp3`, `.wav`, atau `.flac`.

### Human
Audio lebih dekat ke pola yang wajar dibuat atau dinyanyikan manusia.

### Hybrid
Audio berada di area campuran. Bisa jadi:

- sebagian dibuat manusia
- sebagian dibantu AI
- atau produksi modernnya terlalu rapi sehingga hasilnya tidak jelas

### AI
Audio lebih dekat ke pola yang sering muncul pada hasil generator AI.

### Screening
Pemeriksaan awal sebelum diputuskan lebih lanjut.

### Model ML
ML berarti **Machine Learning**.  
Ini adalah model komputer yang belajar dari contoh data.

### Fingerprint
Pola teknis yang sering muncul pada jenis audio tertentu.

---

## 5. Cara menjalankan aplikasi

Cara paling mudah:

1. Buka folder project.
2. Double click file:
   - `AudioAIForensicsLauncher.exe`
3. Browser akan terbuka otomatis.
4. Jika ingin menghentikan aplikasi:
   - double click `AudioAIForensicsStop.exe`

Kalau ingin belajar dari isi project:

- dashboard utama ada di [app.py](D:/My%20Project/Audio_Analyzer/app.py)
- panduan cepat ada di [QUICKSTART.md](D:/My%20Project/Audio_Analyzer/QUICKSTART.md)
- dokumentasi utama ada di [README.md](D:/My%20Project/Audio_Analyzer/README.md)

---

## 6. Cara memakai aplikasi langkah demi langkah

### Langkah 1. Siapkan file audio

Gunakan file seperti:

- `.wav`
- `.mp3`
- `.flac`

### Langkah 2. Upload file audio

Di dashboard, cari bagian upload lalu pilih file.

### Langkah 3. Pilih mode analisis

Ada 3 mode:

#### Mode A: Analisis Ultra-Akurat + Grafik
Cocok kalau ingin melihat detail dan grafik.

#### Mode B: Analisis Ringkas
Cocok kalau ingin hasil lebih cepat dibaca.

#### Mode C: Humanization Editing Focus
Cocok kalau ingin saran perbaikan agar audio tidak terlalu terasa seperti AI.

### Langkah 4. Klik `Jalankan Analisis`

Tunggu proses selesai.

### Langkah 5. Baca hasil dengan pelan

Jangan langsung melihat satu angka saja.  
Lihat juga:

- verdict
- evidence card
- penjelasan awam
- fingerprint

---

## 7. Cara membaca hasil analisis

Biasanya hasil akan menampilkan beberapa bagian:

### A. Headline / Judul Hasil
Contoh:

- `Likely Human`
- `Hybrid / Unclear`
- `AI Suspected (Review)`

### B. Probabilitas
Contoh:

- Human 70%
- Hybrid 20%
- AI 10%

Ini menunjukkan arah hasil.

### C. Forensic Decision
Biasanya berupa:

- `PASS`
- `REVIEW`
- `FAIL`

### D. Prediksi ML
Ini adalah saran dari model machine learning.

### E. Evidence Card
Ini berisi sinyal kuat, sinyal lemah, dan hal-hal yang bisa mirip AI padahal belum tentu AI.

### F. Fingerprint Module
Ini menunjukkan apakah ada pola teknis yang mirip audio generator.

---

## 8. Arti istilah Human, Hybrid, dan AI

### Human
Artinya audio terlihat lebih wajar sebagai hasil manusia.

Tetapi:

- bukan berarti pasti 100% manusia
- hanya berarti sistem lebih condong ke sana

### Hybrid
Artinya hasilnya campuran atau belum jelas.

Ini sering terjadi jika:

- audio dibantu AI
- audio manusia diproses sangat rapi
- ada editing vokal berat
- ada mastering modern yang terlalu stabil

### AI
Artinya sistem melihat lebih banyak pola yang mirip hasil AI.

Tetapi:

- ini tetap harus dibaca bersama evidence dan review manual

---

## 9. Apa itu forensic decision"

Forensic decision adalah keputusan screening utama.

### PASS
Audio cenderung aman dan tidak ada bukti kuat AI.

### REVIEW
Audio perlu dicek manual lagi.

Ini bukan berarti salah.  
Artinya masih ada hal yang perlu dibaca hati-hati.

### FAIL
Ada bukti kuat yang cukup untuk menganggap audio ini sangat patut dicurigai sebagai AI.

Penting:

> `REVIEW` tidak sama dengan `FAIL`.

Banyak siswa bingung di sini.  
`REVIEW` berarti: “tolong cek lagi”.  
Bukan berarti: “pasti AI”.

---

## 10. Apa itu fingerprint module"

Fingerprint module mencoba membaca pola teknis yang kadang mirip audio generator.

Contoh hasil:

- Low
- Medium
- High

### Low
Belum banyak sinyal generator-like.

### Medium
Ada beberapa pola yang mirip, tetapi belum cukup kuat.

### High
Ada cukup banyak pola generator-like.

Tetapi:

> fingerprint tinggi belum otomatis berarti FAIL.

Kenapa"

Karena produksi musik modern juga bisa menimbulkan pola yang mirip.

---

## 11. Contoh membaca hasil dengan bahasa sederhana

Misalnya hasilnya seperti ini:

- Human 17%
- Hybrid 64%
- AI 19%
- Forensic decision: PASS
- Prediksi ML: Hybrid
- Fingerprint: Medium

Cara membacanya:

1. Model melihat audio ini agak campuran.
2. Tapi dari sisi bukti forensik, belum cukup kuat untuk menyatakan AI.
3. Jadi sistem masih memberi `PASS`.
4. Artinya audio lebih aman dibaca sebagai bukan AI keras, walaupun ada pola yang perlu diperhatikan.

Contoh penjelasan awam:

> Audio ini terdengar cukup manusiawi, tetapi ada beberapa karakter produksi yang membuat model membaca hasilnya sebagai campuran. Karena bukti kuat AI belum cukup, sistem belum menganggapnya sebagai audio AI tegas.

---

## 12. Latihan belajar untuk siswa

Supaya lebih paham, coba latihan berikut:

### Latihan 1. Bandingkan 3 audio

Siapkan:

- 1 lagu manusia
- 1 lagu AI
- 1 lagu yang sudah dimixing sangat rapi

Upload satu per satu dan catat hasilnya.

Yang dicatat:

- probabilitas
- forensic decision
- fingerprint
- penjelasan awam

### Latihan 2. Tebak dulu sebelum upload

Sebelum dianalisis, coba tebak:

- menurut kamu ini Human, Hybrid, atau AI"

Lalu bandingkan dengan hasil aplikasi.

### Latihan 3. Cari penyebab REVIEW

Kalau hasilnya `REVIEW`, coba cari:

- apakah vokalnya terlalu rapi"
- apakah musik terlalu stabil"
- apakah dinamikanya terlalu rata"
- apakah bagian chorus dan verse terlalu mirip"

### Latihan 4. Buat catatan pribadi

Buat tabel sederhana:

| Nama file | Tebakan saya | Hasil app | Yang saya pelajari |
|---|---|---|---|

Ini akan sangat membantu proses belajar.

---

## 13. Cara menambah data dan melatih model

Bagian ini lebih cocok untuk belajar tahap lanjutan.

### Menambah data

Setelah hasil analisis keluar, ada tombol:

- `Tambahkan ke Dataset Human`
- `Tambahkan ke Dataset Hybrid`
- `Tambahkan ke Dataset AI`

Gunakan hanya jika kamu **yakin labelnya benar**.

Kenapa"

Karena kalau label salah, model juga akan belajar salah.

### Feature Store

Setelah data diproses, fitur numeriknya bisa disimpan ke:

- `training_features.csv`

Ini berguna agar model tetap bisa dilatih ulang walaupun audio lama sudah dibersihkan.

### Train Ulang

Di dashboard ada panel training untuk:

- sinkron feature store
- train model lagi

Ini cocok untuk belajar bagaimana model berkembang dari waktu ke waktu.

---

## 14. Hal yang boleh dan tidak boleh dilakukan

### Yang boleh

- memakai aplikasi untuk belajar
- membandingkan beberapa file
- membaca hasil dengan kritis
- menambah dataset jika labelnya benar-benar jelas

### Yang tidak boleh

- menuduh orang memakai AI hanya dari satu hasil screening
- menganggap angka probabilitas sebagai kebenaran mutlak
- memasukkan label asal-asalan ke dataset
- memakai hasil ini untuk menyerang orang lain

---

## 15. Pertanyaan yang sering muncul

### Q: Kalau hasilnya AI 80%, apakah pasti AI"

Tidak selalu.  
Artinya sistem sangat curiga, tetapi tetap perlu dibaca bersama evidence dan konteks.

### Q: Kalau hasilnya REVIEW, apakah itu buruk"

Tidak.  
Itu hanya berarti perlu diperiksa lagi.

### Q: Kenapa lagu manusia bisa terbaca mirip AI"

Karena:

- editing vokal berat
- mastering terlalu rata
- musik terlalu repetitif
- produksi terlalu rapi

### Q: Kenapa model bisa berubah setelah training ulang"

Karena model belajar dari data baru yang kamu tambahkan.

### Q: Kenapa data yang salah label berbahaya"

Karena model akan belajar pola yang salah dan hasil berikutnya bisa menyesatkan.

---

## 16. Penutup

Kalau kamu siswa SMA dan sedang belajar dari aplikasi ini, tujuan utamanya bukan sekadar “menebak AI”.

Tujuan belajar yang sebenarnya adalah:

- belajar membaca data
- belajar berpikir kritis
- belajar bahwa model komputer punya keterbatasan
- belajar bahwa hasil teknologi harus dibaca dengan akal sehat

Kalimat terakhir yang paling penting:

> Teknologi yang baik bukan yang selalu terlihat paling pintar, tetapi yang membantu manusia berpikir lebih jernih.

---

## Lampiran Singkat untuk Orang Tua / Pembimbing

Panduan ini cocok dipakai untuk membimbing siswa belajar:

- dasar analisis audio
- cara membaca probabilitas
- perbedaan model ML dan keputusan rule-based
- pentingnya verifikasi manusia
- etika penggunaan teknologi AI

Jika ingin, pembimbing bisa melanjutkan pembelajaran ke tahap:

- membuat dataset yang lebih rapi
- belajar evaluasi model
- memahami false positive dan false negative
- memahami bagaimana sistem screening dibangun

