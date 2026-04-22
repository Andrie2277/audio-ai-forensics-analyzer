import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
import csv
from typing import Optional

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from analyzer_ml import build_feature_vector, compute_dsp_metrics, evaluate_audio, extract_metadata
from ml_pipeline import (
    DEFAULT_FEATURE_STORE_PATH,
    append_feature_row,
    assess_model_reliability,
    assess_prediction_confidence,
    build_feature_store_from_dataset,
    load_model_bundle,
    predict_probabilities,
)
from models import AnalysisReport
from llm_expert import generate_expert_insight, get_minimal_payload
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Audio Forensics & Humanization Analyzer", layout="wide", page_icon="🎧")

def is_demo_mode() -> bool:
    secret_value = None
    try:
        secret_value = st.secrets.get("AUDIO_ANALYZER_DEMO_MODE")
    except Exception:
        secret_value = None
    value = str(secret_value if secret_value is not None else os.getenv("AUDIO_ANALYZER_DEMO_MODE", "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_secret_env(name: str, default: str = "") -> str:
    secret_value = None
    try:
        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None
    value = secret_value if secret_value is not None else os.getenv(name, default)
    return str(value or default).strip()


def render_online_analytics() -> None:
    if st.session_state.get("_aa_analytics_injected"):
        return

    provider = get_secret_env("ANALYTICS_PROVIDER", "").lower()
    snippet = ""

    if provider == "plausible":
        domain = get_secret_env("PLAUSIBLE_DOMAIN", "")
        script_url = get_secret_env("PLAUSIBLE_SCRIPT_URL", "https://plausible.io/js/script.js")
        if domain:
            snippet = f"""
            <script defer data-domain="{domain}" src="{script_url}"></script>
            """
    elif provider == "ga4":
        measurement_id = get_secret_env("GA_MEASUREMENT_ID", "")
        if measurement_id:
            snippet = f"""
            <script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
            <script>
              window.dataLayer = window.dataLayer || [];
              function gtag() {{ dataLayer.push(arguments); }}
              gtag('js', new Date());
              gtag('config', '{measurement_id}', {{'anonymize_ip': true}});
            </script>
            """

    if not snippet:
        return

    try:
        st.html(snippet, unsafe_allow_javascript=True)
    except TypeError:
        st.html(snippet)
    st.session_state["_aa_analytics_injected"] = True


def analytics_provider() -> str:
    return get_secret_env("ANALYTICS_PROVIDER", "").lower()


def analytics_is_enabled() -> bool:
    provider = analytics_provider()
    if provider == "plausible":
        return bool(get_secret_env("PLAUSIBLE_DOMAIN", ""))
    if provider == "ga4":
        return bool(get_secret_env("GA_MEASUREMENT_ID", ""))
    return False


def emit_analytics_event(event_name: str, props: Optional[dict] = None) -> None:
    if not analytics_is_enabled():
        return

    props = props or {}
    event_json = json.dumps(event_name)
    props_json = json.dumps(props)
    provider = analytics_provider()

    if provider == "plausible":
        snippet = f"""
        <script>
          if (window.plausible) {{
            window.plausible({event_json}, {{ props: {props_json} }});
          }}
        </script>
        """
    elif provider == "ga4":
        snippet = f"""
        <script>
          if (window.gtag) {{
            window.gtag('event', {event_json}, {props_json});
          }}
        </script>
        """
    else:
        return

    try:
        st.html(snippet, unsafe_allow_javascript=True)
    except TypeError:
        st.html(snippet)


DEMO_MODE = is_demo_mode()
MODEL_BUNDLE = load_model_bundle()
MODEL_IS_RELIABLE, MODEL_STATUS = assess_model_reliability(MODEL_BUNDLE)
HISTORY_PATH = Path(__file__).with_name("analysis_history.json")
DATASET_ROOT = Path(__file__).with_name("data")
DATASET_CSV_PATH = Path(__file__).with_name("dataset.csv")
FEATURE_STORE_PATH = Path(__file__).with_name(str(DEFAULT_FEATURE_STORE_PATH))
SYNC_REPORT_PATH = Path(__file__).with_name("feature_store_sync_report.json")

TRANSLATIONS = {
    "header_kicker": {"id": "Audio Screening Workspace", "en": "Audio Screening Workspace"},
    "header_title": {"id": "Audio AI Forensics Analyzer", "en": "Audio AI Forensics Analyzer"},
    "header_subtitle": {
        "id": "Dashboard ini dipakai untuk analisis audio, evaluasi hasil forensik, penyimpanan data training, dan retraining model dari feature store tanpa harus menyimpan semua file audio lama.",
        "en": "This dashboard is used for audio analysis, forensic review, training data management, and model retraining from a feature store without keeping all legacy audio files.",
    },
    "language_label": {"id": "Bahasa / Language", "en": "Language / Bahasa"},
    "workspace_summary_title": {"id": "Ringkasan Workspace", "en": "Workspace Summary"},
    "workspace_summary_subtitle": {"id": "Status model, jumlah dataset, feature store, dan history analisis saat ini.", "en": "Current model status, dataset counts, feature store, and analysis history."},
    "model_ready": {"id": "Siap dipakai", "en": "Ready"},
    "model_warn": {"id": "Perlu perhatian", "en": "Needs attention"},
    "feature_store_rows": {"id": "Feature store", "en": "Feature store"},
    "history_items": {"id": "History", "en": "History"},
    "rows": {"id": "baris", "en": "rows"},
    "items": {"id": "item", "en": "items"},
    "total_dataset": {"id": "Total Dataset", "en": "Total Dataset"},
    "human": {"id": "Human", "en": "Human"},
    "hybrid": {"id": "Hybrid", "en": "Hybrid"},
    "ai": {"id": "AI", "en": "AI"},
    "pure_ai": {"id": "Pure AI", "en": "Pure AI"},
    "hint_total_dataset": {"id": "Jumlah file audio yang masih ada di folder data.", "en": "Number of audio files still present in the data folder."},
    "hint_human": {"id": "Sampel berlabel human yang masih tersimpan.", "en": "Stored samples labeled human."},
    "hint_hybrid": {"id": "Sampel berlabel hybrid yang masih tersimpan.", "en": "Stored samples labeled hybrid."},
    "hint_ai": {"id": "Sampel berlabel AI yang masih tersimpan.", "en": "Stored samples labeled AI."},
    "analysis_audio_title": {"id": "Analisis Audio", "en": "Audio Analysis"},
    "analysis_audio_subtitle": {"id": "Upload file, pilih mode, lalu jalankan analisis dari satu tempat.", "en": "Upload a file, choose a mode, then run the analysis from one place."},
    "upload_audio_label": {"id": "Upload Audio File (WAV/MP3/FLAC)", "en": "Upload Audio File (WAV/MP3/FLAC)"},
    "analysis_mode_label": {"id": "Pilih Mode Analisis:", "en": "Choose Analysis Mode:"},
    "mode_a": {"id": "Mode A: Analisis Ultra-Akurat + Grafik", "en": "Mode A: Deep Analysis + Charts"},
    "mode_b": {"id": "Mode B: Analisis Ringkas (Format SubmitHub)", "en": "Mode B: Short Analysis (SubmitHub Format)"},
    "mode_c": {"id": "Mode C: Humanization Editing Focus", "en": "Mode C: Humanization Editing Focus"},
    "run_analysis": {"id": "Jalankan Analisis", "en": "Run Analysis"},
    "quick_flow_title": {"id": "Alur Cepat", "en": "Quick Flow"},
    "quick_flow_subtitle": {"id": "Urutan paling aman untuk kerja harian di dashboard ini.", "en": "The safest daily workflow inside this dashboard."},
    "quick_flow_demo_title": {"id": "Alur Cepat Demo Online", "en": "Online Demo Quick Flow"},
    "quick_flow_demo_subtitle": {"id": "Mode publik hanya untuk analisis audio. Fitur training dan dataset disembunyikan.", "en": "Public mode is limited to audio analysis. Training and dataset features are hidden."},
    "workspace_tab": {"id": "Workspace", "en": "Workspace"},
    "training_tab": {"id": "Training & Dataset", "en": "Training & Dataset"},
    "history_tab": {"id": "History", "en": "History"},
    "how_it_works_tab": {"id": "How It Works", "en": "How It Works"},
    "history_panel": {"id": "History Analisis", "en": "Analysis History"},
    "history_empty": {"id": "Belum ada history analisis.", "en": "No analysis history yet."},
    "open": {"id": "Buka", "en": "Open"},
    "delete": {"id": "Hapus", "en": "Delete"},
    "train_model_title": {"id": "Train Model", "en": "Train Model"},
    "train_model_caption": {"id": "Jalankan training ulang dari training_features.csv. Audio lama boleh dihapus setelah fiturnya tersimpan.", "en": "Retrain the model from training_features.csv. Old audio can be removed after its features are stored."},
    "refresh_dataset_index": {"id": "Refresh Dataset Index", "en": "Refresh Dataset Index"},
    "workspace_public_caption": {"id": "Area publik untuk upload audio, menjalankan analisis, dan membaca hasil screening secara cepat.", "en": "Public area for uploading audio, running analysis, and reading screening results quickly."},
    "workspace_public_info": {"id": "Online demo mode aktif. Training, dataset, dan history persisten disembunyikan agar penggunaan publik tetap aman.", "en": "Online demo mode is active. Training, dataset, and persistent history are hidden to keep public usage safe."},
    "workspace_local_caption": {"id": "Area kerja utama untuk menjalankan analisis dan melihat progres proses saat ini.", "en": "Main workspace for running analysis and viewing current progress."},
    "online_demo_info": {"id": "Mode demo online aktif. Halaman publik ini hanya menampilkan analisis dan penjelasan. Fitur dataset, training, dan pembersihan audio tetap khusus admin/lokal.", "en": "Online demo mode is active. This public page only shows analysis and explanations. Dataset, training, and cleanup features remain admin/local only."},
    "rendered_history_for": {"id": "Menampilkan hasil history untuk", "en": "Showing history result for"},
    "forensic_decision": {"id": "Forensic decision", "en": "Forensic decision"},
    "ml_prediction": {"id": "Prediksi ML", "en": "ML prediction"},
    "headline_probabilities": {"id": "Headline probabilities", "en": "Headline probabilities"},
    "engine": {"id": "Engine", "en": "Engine"},
    "decision_confidence": {"id": "Keyakinan keputusan", "en": "Decision confidence"},
    "fail_confidence": {"id": "Keyakinan untuk FAIL (AI tegas)", "en": "Confidence for FAIL (strong AI)"},
    "headline_source_caption": {"id": "Angka di atas mengikuti basis verdict utama", "en": "The numbers above follow the main verdict basis"},
    "detail_caption_rest": {"id": "Detail di bawah tetap menampilkan heuristik per domain dan keputusan model final.", "en": "The details below still show per-domain heuristics and the final model decision."},
    "main_analysis_tab": {"id": "Main Analysis", "en": "Main Analysis"},
    "ai_auditor_tab": {"id": "AI Auditor", "en": "AI Auditor"},
    "raw_data_tab": {"id": "Raw Data (JSON)", "en": "Raw Data (JSON)"},
    "meaning_of_result": {"id": "Arti Hasil Ini", "en": "What This Result Means"},
    "evidence_card": {"id": "Evidence Card", "en": "Evidence Card"},
    "fingerprint_module": {"id": "Fingerprint Module", "en": "Fingerprint Module"},
    "generator_fingerprint": {"id": "Generator-like fingerprint", "en": "Generator-like fingerprint"},
    "most_consistent_with": {"id": "Most consistent with", "en": "Most consistent with"},
    "fingerprint_confidence": {"id": "Fingerprint confidence", "en": "Fingerprint confidence"},
    "score_breakdown": {"id": "Score breakdown", "en": "Score breakdown"},
    "detail": {"id": "Detail", "en": "Detail"},
    "duration": {"id": "Duration", "en": "Duration"},
    "headline_basis": {"id": "Verdict probabilities (headline basis)", "en": "Verdict probabilities (headline basis)"},
    "source_used": {"id": "Source used", "en": "Source used"},
    "ml_probs_detail": {"id": "ML probabilities (detail transparency)", "en": "ML probabilities (detail transparency)"},
    "engine_used": {"id": "Engine used", "en": "Engine used"},
    "heuristic_combined": {"id": "Heuristic combined probabilities", "en": "Heuristic combined probabilities"},
    "heuristic_spectral": {"id": "Heuristic spectral analysis", "en": "Heuristic spectral analysis"},
    "heuristic_temporal": {"id": "Heuristic temporal analysis", "en": "Heuristic temporal analysis"},
}


def get_language() -> str:
    return st.session_state.get("ui_language", "id")


def t(key: str) -> str:
    lang = get_language()
    return TRANSLATIONS.get(key, {}).get(lang, TRANSLATIONS.get(key, {}).get("id", key))


def ui_text(value: str) -> str:
    if not value:
        return value
    mapping_en = {
        "Likely Human Audio": "Likely Human Audio",
        "Possible AI Detected": "Possible AI Detected",
        "AI Suspected (Review)": "AI Suspected (Review)",
        "Likely Human (Review)": "Likely Human (Review)",
        "Hybrid / Unclear (Review)": "Hybrid / Unclear (Review)",
        "Hybrid Audio / Unclear": "Hybrid Audio / Unclear",
        "PASS": "PASS",
        "REVIEW": "REVIEW",
        "FAIL": "FAIL",
        "Human": "Human",
        "Hybrid": "Hybrid",
        "AI": "AI",
        "Unknown generator": "Unknown generator",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "heuristic_combined": "heuristic_combined",
        "ml_model": "ml_model",
        "ml-model + heuristics": "ml-model + heuristics",
        "heuristics only": "heuristics only",
    }
    mapping_id = {
        "Likely Human Audio": "Audio Cenderung Human",
        "Possible AI Detected": "Kemungkinan AI Terdeteksi",
        "AI Suspected (Review)": "AI Terdeteksi (Review)",
        "Likely Human (Review)": "Cenderung Human (Review)",
        "Hybrid / Unclear (Review)": "Hybrid / Belum Jelas (Review)",
        "Hybrid Audio / Unclear": "Audio Hybrid / Belum Jelas",
        "PASS": "PASS",
        "REVIEW": "REVIEW",
        "FAIL": "FAIL",
        "Human": "Human",
        "Hybrid": "Hybrid",
        "AI": "AI",
        "Unknown generator": "Generator tidak diketahui",
        "low": "rendah",
        "medium": "sedang",
        "high": "tinggi",
        "heuristic_combined": "gabungan_heuristik",
        "ml_model": "model_ml",
        "ml-model + heuristics": "model-ml + heuristik",
        "heuristics only": "heuristik saja",
    }
    return mapping_en.get(value, value) if get_language() == "en" else mapping_id.get(value, value)


def translate_generated_text(text: str) -> str:
    if get_language() != "en" or not text:
        return text

    exact_map = {
        "Tool ini mencoba menilai apakah audio lebih dekat ke pola Human, Hybrid, atau AI. Hasilnya sebaiknya dibaca sebagai screening assistant, bukan alat tuduhan otomatis.": "This tool tries to assess whether the audio is closer to Human, Hybrid, or AI patterns. The result should be read as a screening assistant, not as an automatic accusation tool.",
        "Ini keputusan forensik paling aman. Artinya tidak ada kombinasi bukti kuat yang cukup untuk menuduh AI.": "This is the safest forensic decision. It means there is no strong enough combination of evidence to accuse the file of being AI.",
        "Ini artinya file perlu dicek manual. Ada beberapa hal yang mencurigakan atau tidak stabil, tetapi bukti keras AI belum cukup.": "This means the file still needs manual review. There are suspicious or unstable signs, but strong AI evidence is still insufficient.",
        "Ini artinya ada kombinasi bukti kuat yang cukup serius, jadi file patut dicurigai sebagai AI.": "This means there is a serious enough combination of strong evidence, so the file should be considered suspicious as AI.",
        "Model ML paling condong membaca audio ini sebagai human.": "The ML model leans most strongly toward classifying this audio as human.",
        "Model ML paling condong membaca audio ini sebagai campuran atau berada di area abu-abu antara human dan AI-like pattern.": "The ML model leans most strongly toward classifying this audio as mixed or within the gray area between human and AI-like patterns.",
        "Model ML paling condong membaca audio ini sebagai AI.": "The ML model leans most strongly toward classifying this audio as AI.",
        "Pola fingerprint generator tidak terlalu kuat.": "The generator-like fingerprint is not very strong.",
        "Ada beberapa pola yang mirip audio generator, tetapi masih bisa tercampur dengan produksi modern.": "Some patterns resemble generator audio, but they can still overlap with modern production choices.",
        "Ada cukup banyak pola teknis yang mirip fingerprint audio generator atau audio yang diproses sangat rapi.": "There are quite a few technical patterns that resemble generator fingerprints or extremely polished audio processing.",
        "Tidak ada indikator AI kuat yang aktif pada empat pemeriksaan utama.": "No strong AI indicators are active across the four main checks.",
        "Ada strong evidence yang aktif pada pemeriksaan utama.": "There is active strong evidence in the main checks.",
        "Sebagian temuan lebih cocok dijelaskan oleh produksi modern daripada bukti AI keras.": "Some findings are better explained by modern production choices than by hard AI evidence.",
        "Secara umum audio ini masih terbaca wajar. Sistem tidak menemukan kombinasi bukti kuat yang cukup untuk mencurigai AI.": "Overall, this audio still reads as fairly natural. The system did not find a strong enough combination of evidence to seriously suspect AI.",
        "Ada kombinasi bukti forensik yang cukup kuat, jadi audio ini patut dicurigai sebagai AI atau proses sintetis berat.": "There is a strong enough combination of forensic evidence, so this audio should be suspected as AI or heavily synthetic processing.",
        "Model atau aturan melihat beberapa hal yang mencurigakan, tetapi bukti kuatnya belum cukup untuk vonis keras, jadi hasilnya tetap review manual.": "The model or the rules see some suspicious signs, but the strong evidence is still insufficient for a hard verdict, so the result remains manual review.",
        "Dinamika lagu terlalu rata atau terlalu padat.": "The song's dynamics are too flat or too dense.",
        "Mastering / mix bus": "Mastering / mix bus",
        "Vokal atau nada utama terdengar terlalu rapi dan terlalu stabil.": "The vocal or main melody sounds too polished and too stable.",
        "Track vokal / pitch correction": "Vocal track / pitch correction",
        "Ada indikator teknis yang cukup kuat dan sulit dijelaskan hanya dari mixing biasa.": "There are technical indicators strong enough that they are hard to explain as ordinary mixing alone.",
        "Sumber audio / arrangement / render awal": "Source audio / arrangement / early render",
        "Temuan utama lebih mirip efek mixing atau mastering modern daripada bukti AI yang keras.": "The main findings look more like modern mixing or mastering effects than hard AI evidence.",
        "Mastering, tone shaping, dan kerapian edit": "Mastering, tone shaping, and editing polish",
        "Belum ada masalah besar yang sangat spesifik, tetapi hasilnya masih perlu dibaca hati-hati.": "There is no single very specific major issue yet, but the result still needs to be read carefully.",
        "Review manual pada mix, mastering, dan vokal utama": "Manual review of the mix, mastering, and lead vocal",
    }
    if text in exact_map:
        return exact_map[text]

    replacements = [
        (r"Texture tiling di mel band kuat", "Strong mel-band texture tiling"),
        (r"Texture tiling di mel band cukup jelas", "Clear mel-band texture tiling"),
        (r"Mid band menunjukkan tiling cukup konsisten", "Mid band shows fairly consistent tiling"),
        (r"Air band sangat konsisten antar bagian", "Air band is highly consistent across sections"),
        (r"HF shimmer sangat stasioner", "HF shimmer is highly stationary"),
        (r"HF shimmer cenderung stasioner", "HF shimmer appears fairly stationary"),
        (r"Energi HF tetap kuat saat bagian quiet dibanding loud", "HF energy remains strong in quiet sections compared with loud sections"),
        (r"HF tetap menyala mirip antara bagian quiet dan loud", "HF remains similarly active between quiet and loud sections"),
        (r"Transient band tinggi cenderung seragam", "High-band transients appear rather uniform"),
        (r"Frekuensi tinggi cenderung natural", "High frequencies appear natural"),
        (r"Energi frekuensi tinggi di atas 12kHz sangat menonjol\.", "High-frequency energy above 12 kHz is very prominent."),
        (r"Spectral entropy sangat rendah", "Spectral entropy is very low"),
        (r"Harmonic consistency sangat tinggi dan shimmer HF sangat stasioner\.", "Harmonic consistency is extremely high and HF shimmer is very stationary."),
        (r"Harmonic consistency sangat stabil", "Harmonic consistency is very stable"),
        (r"Rentang dinamis lebar", "Wide dynamic range"),
        (r"Rentang dinamis sempit", "Narrow dynamic range"),
        (r"Dynamic range sempit .* mastering padat\.", "Narrow dynamic range can still come from dense mastering."),
        (r"Variasi pitch sangat stabil", "Pitch variation is very stable"),
        (r"Pitch yang terlalu stabil juga bisa disebabkan AutoTune, Melodyne, atau editing vokal berat\.", "Overly stable pitch can also be caused by AutoTune, Melodyne, or heavy vocal editing."),
        (r"Phase coherence tinggi, tetapi indikator ini masih lemah jika berdiri sendiri\.", "Phase coherence is high, but this indicator remains weak on its own."),
        (r"Phase coherence tinggi bisa muncul dari kompresi, denoise, atau material yang tonal\.", "High phase coherence can come from compression, denoise, or tonal material."),
        (r"Pola 60 detik sangat repetitif dengan jitter drop kecil", "The 60-second pattern is highly repetitive with only a small jitter drop"),
        (r"Pola 60 detik repetitif, tapi masih berubah saat dijitter\.", "The 60-second pattern is repetitive, but still changes when jitter is introduced."),
        (r"Repetisi bagian lagu juga bisa muncul normal pada chorus, loop DAW, atau genre repetitif\.", "Song-part repetition can also appear normally in choruses, DAW loops, or repetitive genres."),
        (r"Ada pola repetitif jangka panjang, tetapi belum cukup kuat untuk menjadi fingerprint AI\.", "There is long-range repetition, but it is not strong enough yet to count as an AI fingerprint."),
        (r"Metadata file mengandung tag yang mengarah ke AI\.", "The file metadata contains tags that point toward AI."),
        (r"S1: Long-range repetition sangat stabil dan tahan jitter, mirip pola tiling\.", "S1: Long-range repetition is highly stable and resists jitter, resembling tiling."),
        (r"S1a aktif: texture tiling pada mel cukup kuat, tetapi belum ada bukti structural tiling dari chroma atau mid band\.", "S1a is active: mel texture tiling is fairly strong, but there is still no structural tiling evidence from chroma or mid band."),
        (r"S1b aktif: ada structural tiling ringan pada chroma atau mid band, tetapi texture tiling global belum cukup kuat\.", "S1b is active: there is mild structural tiling in chroma or mid band, but global texture tiling is still not strong enough."),
        (r"Ada repetisi jangka panjang, tetapi belum cukup kuat atau belum cukup tahan jitter untuk disebut tiling\.", "There is long-range repetition, but it is not yet strong enough or jitter-resistant enough to be called tiling."),
        (r"S2: HF shimmer sangat stasioner dan air band terlalu konsisten antar bagian\.", "S2: HF shimmer is highly stationary and the air band is too consistent across sections."),
        (r"S2 weak aktif: air band cukup stabil, tetapi belum didukung oleh over_0.98 ratio yang kuat atau shimmer stationarity yang cukup\.", "S2 weak is active: the air band is fairly stable, but it is not yet supported by a strong over_0.98 ratio or enough shimmer stationarity."),
        (r"Air band cukup stabil, tetapi sendirian belum cukup kuat untuk menuduh AI\.", "The air band is fairly stable, but alone it is still not strong enough to accuse AI."),
        (r"S3: Perilaku vokal sangat terkunci pada pitch proxy yang tersedia\.", "S3: Vocal behavior appears strongly locked according to the available pitch proxies."),
        (r"Ada indikasi pitch vokal sangat rapi, tetapi bukti vocal synthesis belum lengkap\.", "There are signs of very tidy vocal pitch, but the evidence for vocal synthesis is still incomplete."),
        (r"S4: Metadata file mengandung tag yang mengarah ke AI\.", "S4: The file metadata contains tags that point toward AI."),
        (r"Energi 8-12 kHz cukup tinggi; ini bisa datang dari exciter, de-esser, hi-hat, atau sibilance\.", "Energy in the 8–12 kHz range is fairly high; this can come from exciter, de-esser, hi-hat, or sibilance."),
        (r"Sebagian MFCC mengarah ke karakter bright/air, tetapi ini lebih cocok dibaca sebagai warna produksi\.", "Some MFCC values point toward a bright/airy character, but this is better read as production color."),
        (r"Model ML cukup curiga ke AI, tetapi bukti forensik kuat masih belum cukup\.", "The ML model is fairly suspicious of AI, but the strong forensic evidence is still insufficient."),
        (r"Model lebih melihat risiko hybrid karena ciri produksi modern cukup dominan\.", "The model sees more hybrid risk because modern production traits are fairly dominant."),
        (r"Durasi di luar rentang 32-180 detik cenderung membuat analisis kurang stabil\.", "Duration outside the 32–180 second range tends to make the analysis less stable."),
        (r"Pada durasi panjang, pola repetisi harus dibaca lebih hati-hati karena struktur lagu normal juga bisa tampak mirip antar bagian\.", "For long durations, repetition patterns must be read more carefully because normal song structure can also look similar across sections."),
        (r"Kasus hybrid masih paling rentan tertukar dengan mixing modern atau editing vokal berat\.", "Hybrid cases remain the most vulnerable to being confused with modern mixing or heavy vocal editing."),
        (r"Keputusan forensik saat ini adalah", "The current forensic decision is"),
        (r"dengan verdict", "with verdict"),
        (r"Prediksi ML paling condong ke kelas", "The ML prediction leans most toward class"),
        (r"Strong evidence: ", "Strong evidence: "),
        (r"Tidak ada strong evidence yang cukup untuk FAIL, tetapi fingerprint generator-like terdeteksi \(tingkat: ([^)]+)\)\.", r"There is not enough strong evidence to justify FAIL, but a generator-like fingerprint is detected (level: \1)."),
        (r"Weak/context evidence:", "Weak/context evidence:"),
        (r"sinyal lemah", "weak signals"),
        (r"Produksi modern dengan kompresi berat, AutoTune, atau MIDI yang sangat rapi bisa terbaca mirip AI\.", "Heavy modern compression, AutoTune, or very tidy MIDI can read as AI-like."),
        (r"Kurangi limiter atau kompresor yang terlalu menekan di master\.", "Reduce overly aggressive limiter or compressor settings on the master."),
        (r"Biarkan verse dan chorus punya perbedaan energi yang lebih terasa\.", "Let the verse and chorus have a more noticeable energy difference."),
        (r"Cek apakah loudness terlalu dipaksa rata dari awal sampai akhir\.", "Check whether the loudness has been forced too flat from beginning to end."),
        (r"Longgarkan AutoTune atau retune speed\.", "Loosen AutoTune or retune speed."),
        (r"Jangan ratakan semua nada secara terlalu presisi\.", "Do not flatten every note too precisely."),
        (r"Biarkan sedikit gerakan alami pada pitch dan timing vokal\.", "Allow a little natural movement in vocal pitch and timing."),
        (r"Bandingkan dengan stem asli atau premaster tanpa limiter\.", "Compare against the original stems or a pre-master without limiting."),
        (r"Cek apakah ada bagian yang terlalu copy-paste atau terlalu identik antar segmen\.", "Check whether any sections feel too copy-pasted or too identical across segments."),
        (r"Pastikan sumber vokal dan instrumen benar-benar berasal dari take atau aransemen yang berbeda\.", "Make sure the vocal and instrumental sources truly come from different takes or arrangements."),
        (r"Kurangi brightening atau exciter berlebih di area atas\.", "Reduce excessive brightening or exciter in the upper range."),
        (r"Cek apakah limiter, de-esser, atau denoise terlalu agresif\.", "Check whether the limiter, de-esser, or denoise is too aggressive."),
        (r"Beri variasi kecil antar bagian agar tidak terasa terlalu seragam\.", "Add small variations across sections so they do not feel too uniform."),
        (r"Dengarkan ulang bagian verse, chorus, dan transisi untuk mencari bagian yang terlalu seragam\.", "Listen again to the verse, chorus, and transitions to find sections that feel too uniform."),
        (r"Bandingkan versi master dengan versi yang belum terlalu diproses jika ada\.", "Compare the master version with a less processed version if available."),
        (r"Prioritaskan pemeriksaan dinamika, vokal utama, dan variasi antar bagian lagu\.", "Prioritize checking dynamics, the lead vocal, and variation across song sections."),
    ]

    translated = text
    for pattern, repl in replacements:
        translated = re.sub(pattern, repl, translated)
    return translated


def translate_generated_list(items: list[str]) -> list[str]:
    return [translate_generated_text(item) for item in items]


def plot_waveform_rms(y, sr):
    times = librosa.frames_to_time(np.arange(len(y) // 512), sr=sr, hop_length=512)
    rms = librosa.feature.rms(y=y, hop_length=512)[0]

    skip = max(1, len(times) // 1000)
    times_plot = times[::skip]
    rms_plot = rms[::skip]
    y_plot_times = np.linspace(0, len(y) / sr, num=1000)
    y_plot = y[np.linspace(0, len(y) - 1, num=1000).astype(int)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_plot_times, y=y_plot, mode="lines", name="Waveform", line=dict(color="rgba(135,206,235,0.4)")))
    fig.add_trace(go.Scatter(x=times_plot, y=rms_plot, mode="lines", name="RMS Envelope", line=dict(color="yellow")))
    fig.update_layout(
        title="Waveform & RMS Energy (Temporal Analysis)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_dark",
    )
    return fig


def plot_f0(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    times = librosa.times_like(f0, sr=sr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=f0, mode="markers", name="F0 Contour", marker=dict(color="cyan", size=3)))
    fig.update_layout(
        title="Pitch (F0) Tracking (Temporal Analysis)",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_dark",
        yaxis_type="log",
    )
    return fig


def plot_spectrum(y, sr):
    spectrum = np.abs(librosa.stft(y))
    mean_spectrum = np.mean(spectrum, axis=1)
    freqs = librosa.fft_frequencies(sr=sr)
    mean_spectrum_db = librosa.amplitude_to_db(mean_spectrum, ref=np.max)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=mean_spectrum_db, mode="lines", name="Avg Spectrum", line=dict(color="magenta")))
    fig.add_vrect(x0=12000, x1=20000, fillcolor="red", opacity=0.2, line_width=0, annotation_text="AI Shimmer Zone")
    fig.update_layout(
        title="Global Average Spectrum (Spectral Analysis)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        template="plotly_dark",
    )
    fig.update_xaxes(type="log", range=[np.log10(20), np.log10(sr / 2)])
    return fig


def load_history():
    if DEMO_MODE:
        return st.session_state.setdefault("_demo_history_items", [])
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(history_items):
    if DEMO_MODE:
        st.session_state["_demo_history_items"] = history_items
        return
    HISTORY_PATH.write_text(json.dumps(history_items, ensure_ascii=False, indent=2), encoding="utf-8")


def load_sync_report() -> Optional[dict]:
    if not SYNC_REPORT_PATH.exists():
        return None
    try:
        return json.loads(SYNC_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_sync_report(report: dict) -> None:
    SYNC_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def inject_custom_css():
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light !important;
        }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp {
            color-scheme: light !important;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(26, 93, 73, 0.12), transparent 26%),
                radial-gradient(circle at top right, rgba(177, 111, 55, 0.10), transparent 22%),
                linear-gradient(180deg, #f7f6f1 0%, #fbfaf7 100%);
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(26, 93, 73, 0.12), transparent 26%),
                radial-gradient(circle at top right, rgba(177, 111, 55, 0.10), transparent 22%),
                linear-gradient(180deg, #f7f6f1 0%, #fbfaf7 100%) !important;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }
        header[data-testid="stHeader"] [data-testid="stToolbar"] {
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
        header[data-testid="stHeader"] [data-testid="stToolbar"]::before {
            content: "🎧";
            font-size: 1.25rem;
            line-height: 1;
            margin-right: 0.15rem;
        }
        header[data-testid="stHeader"] [data-testid="stToolbar"] > div:first-child svg,
        header[data-testid="stHeader"] [data-testid="stToolbar"] > button:first-child svg {
            display: none !important;
        }
        .main .block-container,
        .main .block-container p,
        .main .block-container li,
        .main .block-container label,
        .main .block-container span,
        .main .block-container div,
        .main .block-container h1,
        .main .block-container h2,
        .main .block-container h3,
        .main .block-container h4,
        .main .block-container h5,
        .main .block-container h6 {
            color: #1f2924;
        }
        .main .block-container a {
            color: #1f6feb !important;
        }
        .aa-hero {
            padding: 1.5rem 1.6rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #123227 0%, #1f4d3d 48%, #d9c7a5 100%);
            color: #fbfaf7;
            box-shadow: 0 18px 50px rgba(18, 50, 39, 0.14);
            margin-bottom: 1rem;
        }
        .aa-kicker {
            font-size: 0.82rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            opacity: 0.76;
            margin-bottom: 0.45rem;
        }
        .aa-title {
            font-size: 2.35rem;
            font-weight: 700;
            line-height: 1.05;
            margin-bottom: 0.6rem;
        }
        .aa-subtitle {
            font-size: 1rem;
            line-height: 1.6;
            max-width: 780px;
            opacity: 0.92;
        }
        .aa-hero,
        .aa-hero div,
        .aa-hero span,
        .aa-hero p,
        .aa-hero h1,
        .aa-hero h2,
        .aa-hero h3 {
            color: #fbfaf7 !important;
        }
        .aa-panel {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(18, 50, 39, 0.08);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 30px rgba(19, 34, 27, 0.06);
            margin-bottom: 1rem;
        }
        .aa-panel-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #183128;
            margin-bottom: 0.25rem;
        }
        .aa-panel-subtitle {
            color: #63736a;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .aa-metric-card {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(18, 50, 39, 0.08);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 24px rgba(19, 34, 27, 0.05);
        }
        .aa-metric-label {
            color: #6c7a71;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        }
        .aa-metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #183128;
            line-height: 1.1;
        }
        .aa-metric-hint {
            color: #6c7a71;
            font-size: 0.84rem;
            margin-top: 0.35rem;
            line-height: 1.35;
        }
        .aa-badge {
            display: inline-block;
            padding: 0.32rem 0.6rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-bottom: 0.35rem;
            background: rgba(18, 50, 39, 0.08);
            color: #183128;
        }
        .aa-badge.good { background: rgba(32, 119, 86, 0.14); color: #176d50; }
        .aa-badge.warn { background: rgba(185, 120, 54, 0.15); color: #9b5f22; }
        .aa-badge.soft { background: rgba(67, 92, 149, 0.10); color: #314f91; }
        .aa-steps {
            margin: 0;
            padding-left: 1.2rem;
            color: #425148;
        }
        .aa-steps li {
            margin-bottom: 0.45rem;
        }
        div[data-testid="stRadio"] label,
        div[data-testid="stFileUploader"] label,
        div[data-testid="stTabs"] button,
        div[data-testid="stTabs"] button p,
        div[data-testid="stExpander"] summary,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stCaptionContainer"] {
            color: #1f2924 !important;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div {
            color: #1f2924 !important;
        }
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.96) !important;
            color: #183128 !important;
            border: 1px solid rgba(18, 50, 39, 0.12) !important;
            box-shadow: 0 8px 24px rgba(19, 34, 27, 0.06) !important;
        }
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * {
            color: #183128 !important;
        }
        [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button {
            background: #f6f3eb !important;
            color: #183128 !important;
            border: 1px solid rgba(18, 50, 39, 0.14) !important;
        }
        [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button:hover,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button:hover {
            background: #ece7db !important;
            color: #183128 !important;
        }
        [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"],
        [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] * {
            color: #f8faf8 !important;
        }
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
        [data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"] {
            color: #1f2924 !important;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] [data-testid="stCaptionContainer"],
        [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
            color: #5b6761 !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label *,
        div[data-testid="stButton"] button *,
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSelectbox"] div,
        div[data-testid="stTextInput"] label,
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stNumberInput"] input {
            color: #1f2924 !important;
        }
        div[data-testid="stButton"] button {
            color: #1f2924 !important;
        }
        .aa-article {
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(18, 50, 39, 0.08);
            border-radius: 22px;
            padding: 1.35rem 1.4rem;
            box-shadow: 0 10px 30px rgba(19, 34, 27, 0.06);
            color: #1f2924;
        }
        .aa-article h1,
        .aa-article h2,
        .aa-article h3,
        .aa-article p,
        .aa-article li,
        .aa-article div,
        .aa-article span {
            color: #1f2924 !important;
        }
        .aa-article .stAlert,
        .aa-article [data-testid="stAlert"] {
            color: inherit;
        }
        .aa-article code {
            color: #163128 !important;
            background: rgba(18, 50, 39, 0.06);
            padding: 0.1rem 0.35rem;
            border-radius: 8px;
        }
        .aa-callout {
            background: linear-gradient(135deg, rgba(24, 49, 40, 0.96), rgba(34, 82, 65, 0.96));
            color: #f8faf8 !important;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin: 0.8rem 0 1rem 0;
            box-shadow: 0 12px 24px rgba(18, 50, 39, 0.12);
        }
        .aa-callout * {
            color: #f8faf8 !important;
        }
        .aa-callout-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .aa-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.9rem;
            margin: 0.9rem 0 1.1rem 0;
        }
        .aa-summary-card {
            background: rgba(247, 246, 241, 0.98);
            border: 1px solid rgba(18, 50, 39, 0.10);
            border-radius: 18px;
            padding: 0.95rem 1rem;
        }
        .aa-summary-card-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #183128 !important;
            margin-bottom: 0.35rem;
        }
        .aa-summary-card p {
            margin: 0;
            color: #415148 !important;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_dataset_summary() -> dict:
    counts = {"human": 0, "hybrid": 0, "ai": 0}
    for file_path in get_dataset_audio_paths():
        label = file_path.parent.name.lower()
        if label in counts:
            counts[label] += 1
    counts["total"] = sum(counts.values())
    return counts


def render_metric_card(label: str, value: str, hint: str = ""):
    st.markdown(
        f"""
        <div class="aa-metric-card">
            <div class="aa-metric-label">{label}</div>
            <div class="aa-metric-value">{value}</div>
            <div class="aa-metric-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_header():
    inject_custom_css()
    selector_col1, selector_col2 = st.columns([5, 1.2])
    with selector_col2:
        selected_language = st.selectbox(
            t("language_label"),
            options=["Bahasa Indonesia", "English"],
            index=0 if get_language() == "id" else 1,
            key="language_selector",
        )
        st.session_state["ui_language"] = "id" if selected_language == "Bahasa Indonesia" else "en"
    st.markdown(
        f"""
        <div class="aa-hero">
            <div class="aa-kicker">{t("header_kicker")}</div>
            <div class="aa-title">{t("header_title")}</div>
            <div class="aa-subtitle">
                {t("header_subtitle")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_overview():
    dataset_summary = get_dataset_summary()
    history_count = len(load_history())
    reliability_badge = "good" if MODEL_IS_RELIABLE else "warn"
    reliability_text = t("model_ready") if MODEL_IS_RELIABLE else t("model_warn")

    st.markdown(
        f"""
        <div class="aa-panel">
            <div class="aa-panel-title">{t("workspace_summary_title")}</div>
            <div class="aa-panel-subtitle">{t("workspace_summary_subtitle")}</div>
            <div style="margin-top:0.8rem;">
                <span class="aa-badge {reliability_badge}">Model: {reliability_text}</span>
                <span class="aa-badge soft">{t("feature_store_rows")}: {get_feature_store_count()} {t("rows")}</span>
                <span class="aa-badge warn">{t("history_items")}: {history_count} {t("items")}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(t("total_dataset"), str(dataset_summary["total"]), t("hint_total_dataset"))
    with col2:
        render_metric_card(t("human"), str(dataset_summary["human"]), t("hint_human"))
    with col3:
        render_metric_card(t("hybrid"), str(dataset_summary["hybrid"]), t("hint_hybrid"))
    with col4:
        render_metric_card(t("ai"), str(dataset_summary["ai"]), t("hint_ai"))


def add_history_entry(report: AnalysisReport, source_name: str, source_type: str, reference_link: str):
    history_items = load_history()
    entry_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    history_items.insert(
        0,
        {
            "id": entry_id,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "source_name": source_name,
            "source_type": source_type,
            "reference_link": reference_link,
            "prediction": report.overall_verdict,
            "screening_outcome": report.screening_outcome,
            "report": report.model_dump(),
        },
    )
    save_history(history_items[:30])
    return entry_id


def format_relative_time(iso_string: str) -> str:
    try:
        created = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
    except Exception:
        return iso_string
    delta = datetime.utcnow().replace(tzinfo=created.tzinfo) - created
    seconds = int(delta.total_seconds())
    is_en = get_language() == "en"
    if seconds < 60:
        return "just now" if is_en else "baru saja"
    if seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minutes ago" if is_en else f"{minutes} menit lalu"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hours ago" if is_en else f"{hours} jam lalu"
    days = seconds // 86400
    return f"{days} days ago" if is_en else f"{days} hari lalu"


def sanitize_filename(filename: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in (" ", ".", "_", "-") else "_" for ch in filename).strip()
    return cleaned or "uploaded_audio.wav"


def rebuild_dataset_csv():
    rows = []
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4"}
    for label in ("human", "hybrid", "ai"):
        label_dir = DATASET_ROOT / label
        if not label_dir.exists():
            continue
        for file_path in sorted(label_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
                rows.append({"path": file_path.relative_to(Path(__file__).parent).as_posix(), "label": label})

    with DATASET_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)


def add_uploaded_audio_to_dataset(target_label: str, uploaded_name: str, uploaded_bytes: bytes) -> Path:
    target_dir = DATASET_ROOT / target_label
    target_dir.mkdir(parents=True, exist_ok=True)
    base_name = sanitize_filename(uploaded_name)
    candidate = target_dir / base_name
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while candidate.exists():
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    candidate.write_bytes(uploaded_bytes)
    rebuild_dataset_csv()
    return candidate


def save_training_features(target_label: str, report: AnalysisReport, saved_path: Path) -> None:
    append_feature_row(
        feature_store_csv=str(FEATURE_STORE_PATH),
        label=target_label,
        feature_vector=report.feature_vector,
        source_name=report.metadata.filename,
        source_path=str(saved_path.relative_to(Path(__file__).parent).as_posix()),
    )


def get_feature_store_count() -> int:
    if not FEATURE_STORE_PATH.exists():
        return 0
    try:
        with FEATURE_STORE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return sum(1 for _ in reader)
    except Exception:
        return 0


def get_feature_store_source_paths() -> set[str]:
    if not FEATURE_STORE_PATH.exists():
        return set()
    try:
        with FEATURE_STORE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return {
                (row.get("source_path") or "").strip().replace("\\", "/").lower()
                for row in reader
                if (row.get("source_path") or "").strip()
            }
    except Exception:
        return set()


def get_dataset_audio_paths() -> list[Path]:
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4"}
    files = []
    for label in ("human", "hybrid", "ai"):
        label_dir = DATASET_ROOT / label
        if not label_dir.exists():
            continue
        for file_path in sorted(label_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
                files.append(file_path)
    return files


def get_safe_delete_candidates() -> tuple[list[Path], list[Path]]:
    feature_paths = get_feature_store_source_paths()
    dataset_files = get_dataset_audio_paths()
    safe_files = []
    unsafe_files = []
    for file_path in dataset_files:
        relative_path = file_path.relative_to(Path(__file__).parent).as_posix().lower()
        if relative_path in feature_paths:
            safe_files.append(file_path)
        else:
            unsafe_files.append(file_path)
    return safe_files, unsafe_files


def delete_dataset_files(paths: list[Path]) -> dict:
    deleted = []
    failed = []
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                deleted.append(str(path))
        except Exception as exc:
            failed.append({"path": str(path), "reason": str(exc)})
    rebuild_dataset_csv()
    return {"deleted": deleted, "failed": failed}


def normalize_skip_reason(reason: str) -> str:
    cleaned = (reason or "").strip()
    if not cleaned:
        return "Gagal dibaca oleh decoder audio. File kemungkinan perlu dikonversi ke WAV/MP3 yang lebih bersih."
    return cleaned


def render_progress_block(container, percent: int, processed: int, total: int, added: int, skipped: int, current_file: str, title: str):
    with container.container():
        st.markdown(f"### {percent}%")
        st.progress(min(max(percent, 0), 100))
        col1, col2, col3 = st.columns(3)
        col1.metric("File", f"{processed}/{total}")
        col2.metric("Success" if get_language() == "en" else "Berhasil", str(added))
        col3.metric("Skipped" if get_language() == "en" else "Dilewati", str(skipped))
        if current_file:
            st.caption(f"{title}: `{current_file}`")


def render_analysis_progress(container, percent: int, step_label: str, filename: str):
    with container.container():
        st.markdown(f"### {percent}%")
        st.progress(min(max(percent, 0), 100))
        col1, col2 = st.columns(2)
        col1.metric("Progress", f"{percent}%")
        col2.metric("Stage" if get_language() == "en" else "Tahap", step_label)
        st.caption(f"{'Processing file' if get_language() == 'en' else 'File yang diproses'}: `{filename}`")


def render_skipped_files(report: Optional[dict]):
    if not report:
        return
    skipped_rows = report.get("rows_skipped") or []
    existing_skipped_rows = [item for item in skipped_rows if Path(item.get("path", "")).exists()]
    skipped_rows = existing_skipped_rows
    if not skipped_rows:
        return

    if get_language() == "en":
        st.warning(f"{len(skipped_rows)} files were skipped. These files should be converted to WAV and synced again.")
    else:
        st.warning(f"Ada {len(skipped_rows)} file yang dilewati. File ini sebaiknya dikonversi ke WAV lalu disinkron ulang.")
    with st.expander("Files that should be converted" if get_language() == "en" else "Daftar file yang perlu dikonversi", expanded=True):
        for item in skipped_rows:
            st.markdown(f"- `{item.get('path', '-')}`")
            st.caption(f"{'Reason' if get_language() == 'en' else 'Alasan'}: {normalize_skip_reason(item.get('reason', ''))}")


render_dashboard_header()
if DEMO_MODE:
    render_online_analytics()
render_workspace_overview()

control_col, ops_col = st.columns([1.35, 0.85], gap="large")
with control_col:
    analysis_title = t("analysis_audio_title")
    analysis_subtitle = t("analysis_audio_subtitle")
    st.markdown(
        f"""
        <div class="aa-panel">
            <div class="aa-panel-title">{analysis_title}</div>
            <div class="aa-panel-subtitle">{analysis_subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(t("upload_audio_label"), type=["wav", "mp3", "flac", "ogg"])
    mode = st.radio(
        t("analysis_mode_label"),
        ["A", "B", "C"],
        format_func=lambda value: {
            "A": t("mode_a"),
            "B": t("mode_b"),
            "C": t("mode_c"),
        }[value],
    )
    analyze_clicked = st.button(t("run_analysis"), disabled=uploaded_file is None, use_container_width=True)

    current_upload_name = uploaded_file.name if uploaded_file is not None else ""
    previous_upload_name = st.session_state.get("_analytics_last_upload_name", "")
    if current_upload_name and current_upload_name != previous_upload_name:
        emit_analytics_event(
            "upload_audio",
            {
                "filename": current_upload_name,
                "extension": Path(current_upload_name).suffix.lower(),
                "mode": mode,
            },
        )
    st.session_state["_analytics_last_upload_name"] = current_upload_name

    previous_mode = st.session_state.get("_analytics_last_mode")
    if previous_mode is not None and previous_mode != mode:
        emit_analytics_event(
            "change_mode",
            {
                "from": previous_mode,
                "to": mode,
            },
        )
    st.session_state["_analytics_last_mode"] = mode

    if analyze_clicked and uploaded_file is not None:
        emit_analytics_event(
            "run_analysis",
            {
                "mode": mode,
                "filename": uploaded_file.name,
                "extension": Path(uploaded_file.name).suffix.lower(),
            },
        )

with ops_col:
    if DEMO_MODE:
        st.markdown(
            f"""
            <div class="aa-panel">
                <div class="aa-panel-title">{t("quick_flow_demo_title")}</div>
                <div class="aa-panel-subtitle">{t("quick_flow_demo_subtitle")}</div>
                <ol class="aa-steps">
                    <li>{'Upload audio from the browser.' if get_language() == 'en' else 'Upload audio dari browser.'}</li>
                    <li>{'Choose the analysis mode you want to use.' if get_language() == 'en' else 'Pilih mode analisis yang ingin dipakai.'}</li>
                    <li>{'Click <b>Run Analysis</b> and read the main result.' if get_language() == 'en' else 'Klik <b>Jalankan Analisis</b> lalu baca hasil utamanya.'}</li>
                    <li>{'Use the result as a screening assistant, not as an automatic final decision.' if get_language() == 'en' else 'Gunakan hasil sebagai screening assistant, bukan keputusan final otomatis.'}</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="aa-panel">
                <div class="aa-panel-title">{t("quick_flow_title")}</div>
                <div class="aa-panel-subtitle">{t("quick_flow_subtitle")}</div>
                <ol class="aa-steps">
                    <li>{'Upload audio and run the analysis.' if get_language() == 'en' else 'Upload audio lalu jalankan analisis.'}</li>
                    <li>{'If the label is already confirmed, save the file to the correct dataset.' if get_language() == 'en' else 'Jika label file sudah yakin, simpan ke dataset yang benar.'}</li>
                    <li>{'Sync the feature store to preserve numerical knowledge.' if get_language() == 'en' else 'Sinkronkan feature store untuk mengamankan pengetahuan numerik.'}</li>
                    <li>{'Train the model from the feature store.' if get_language() == 'en' else 'Train model dari feature store.'}</li>
                    <li>{'Delete old audio that is already safe to remove if you want to tidy the folder.' if get_language() == 'en' else 'Hapus audio lama yang sudah aman jika ingin merapikan folder.'}</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_history_panel():
    history_items = load_history()
    with st.expander(t("history_panel"), expanded=False):
        if not history_items:
            st.caption(t("history_empty"))
            return
        for entry in history_items:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                st.markdown(f"**{entry['source_type']}**")
                st.caption(entry["source_name"])
                if entry.get("reference_link"):
                    st.caption(entry["reference_link"])
            with col2:
                st.caption(format_relative_time(entry["created_at"]))
                st.caption(entry["screening_outcome"])
            with col3:
                st.markdown(entry["prediction"])
            with col4:
                if st.button(t("open"), key=f"open_{entry['id']}"):
                    st.session_state["selected_history_id"] = entry["id"]
                if st.button(t("delete"), key=f"delete_{entry['id']}"):
                    remaining = [item for item in history_items if item["id"] != entry["id"]]
                    save_history(remaining)
                    if st.session_state.get("selected_history_id") == entry["id"]:
                        st.session_state["selected_history_id"] = None
                    st.rerun()


def render_training_panel():
    with st.expander(t("train_model_title"), expanded=False):
        st.caption(t("train_model_caption"))
        st.caption(f"{t('feature_store_rows')}: {FEATURE_STORE_PATH.name}")
        st.caption(f"{'Current feature store rows' if get_language() == 'en' else 'Jumlah baris feature store saat ini'}: {get_feature_store_count()}")
        top_actions_col1, top_actions_col2 = st.columns([1, 3])
        with top_actions_col1:
            if st.button(t("refresh_dataset_index"), key="refresh_dataset_index_button"):
                rebuild_dataset_csv()
                st.success("dataset.csv refreshed from the data folder." if get_language() == "en" else "dataset.csv berhasil disinkronkan ulang dari isi folder data.")
                st.rerun()
        st.markdown("**Usage steps:**" if get_language() == "en" else "**Langkah penggunaan:**")
        st.markdown("1. If you still have old audio in the `data` folder, click `Sync Feature Store from Existing Dataset` once first." if get_language() == "en" else "1. Jika kamu masih punya audio lama di folder `data`, klik `Sinkronkan Feature Store dari Dataset Lama` sekali dulu.")
        st.markdown("2. After syncing finishes, the feature store will preserve numerical knowledge from the old files." if get_language() == "en" else "2. Setelah sinkron selesai, feature store akan menyimpan pengetahuan numerik dari file lama.")
        st.markdown("3. For new files, analyze them first, then click `Add to Human/Hybrid/AI Dataset`." if get_language() == "en" else "3. Untuk file baru, analisis dulu lalu klik `Tambahkan ke Dataset Human/Hybrid/AI`.")
        st.markdown("4. After a few new files have been added, click `Train Model from feature store`." if get_language() == "en" else "4. Setelah beberapa data baru masuk, klik `Train Model dari feature store`.")
        st.markdown("5. Once training is complete and the feature store is safe, old audio may be deleted if you no longer want to keep it." if get_language() == "en" else "5. Setelah training selesai dan feature store aman, audio lama boleh dihapus kalau memang tidak ingin disimpan.")
        if st.button("Sync Feature Store from Existing Dataset" if get_language() == "en" else "Sinkronkan Feature Store dari Dataset Lama", key="sync_feature_store_button"):
            progress_container = st.empty()
            status_placeholder = st.empty()

            def on_progress(payload: dict):
                total = max(int(payload.get("total", 0)), 1)
                processed = int(payload.get("processed", 0))
                added = int(payload.get("added", 0))
                skipped = int(payload.get("skipped", 0))
                current_file = payload.get("current_file", "")
                percent = int((processed / total) * 100)
                render_progress_block(
                    progress_container,
                    percent=percent,
                    processed=processed,
                    total=total,
                    added=added,
                    skipped=skipped,
                    current_file=current_file,
                    title="Processing" if get_language() == "en" else "Sedang diproses",
                )
                if payload.get("status") == "skipped":
                    status_placeholder.caption(
                        f"{'Skipped file' if get_language() == 'en' else 'File dilewati'}: `{current_file}`. {'Reason' if get_language() == 'en' else 'Alasan'}: {normalize_skip_reason(payload.get('reason', ''))}"
                    )

            result = build_feature_store_from_dataset(
                dataset_csv=str(DATASET_CSV_PATH),
                feature_store_csv=str(FEATURE_STORE_PATH),
                progress_callback=on_progress,
            )
            save_sync_report(result)
            st.session_state["last_sync_report"] = result
            st.success(
                f"Sync complete. Added {result['rows_added']} feature rows from {result['rows_seen']} dataset files." if get_language() == "en"
                else f"Sinkronisasi selesai. Ditambahkan {result['rows_added']} baris fitur dari {result['rows_seen']} file dataset."
            )
            render_skipped_files(result)
        latest_sync_report = st.session_state.get("last_sync_report") or load_sync_report()
        render_skipped_files(latest_sync_report)

        safe_files, unsafe_files = get_safe_delete_candidates()
        skipped_paths = {item.get("path", "") for item in (latest_sync_report or {}).get("rows_skipped", [])}
        skipped_unsafe_files = [path for path in unsafe_files if str(path) in skipped_paths]

        st.markdown("**Legacy Audio Cleanup**" if get_language() == "en" else "**Pembersihan Audio Lama**")
        st.caption(
            "Only files already recorded in the feature store are safe to delete. Skipped files will not be deleted."
            if get_language() == "en"
            else "Hanya file yang sudah tercatat di feature store yang aman dihapus. File skipped tidak akan ikut dihapus."
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Safe to delete" if get_language() == "en" else "Aman dihapus", str(len(safe_files)))
        col2.metric("Not safe yet" if get_language() == "en" else "Belum aman", str(len(unsafe_files)))
        col3.metric("Skipped", str(len(skipped_unsafe_files)))

        if safe_files:
            with st.expander("View files that are safe to delete" if get_language() == "en" else "Lihat daftar file yang aman dihapus", expanded=False):
                preview_safe = safe_files[:20]
                for path in preview_safe:
                    st.markdown(f"- `{path}`")
                if len(safe_files) > len(preview_safe):
                    st.caption(
                        f"There are still {len(safe_files) - len(preview_safe)} more files that are also safe to delete."
                        if get_language() == "en"
                        else f"Masih ada {len(safe_files) - len(preview_safe)} file lain yang juga aman dihapus."
                    )

            confirm_delete = st.checkbox(
                "I confirm that I want to delete legacy audio files that are already safe in the feature store"
                if get_language() == "en"
                else "Saya yakin ingin menghapus file audio lama yang sudah aman di feature store",
                key="confirm_safe_delete",
            )
            if st.button("Delete Safe Legacy Audio" if get_language() == "en" else "Hapus Audio Lama yang Sudah Aman", key="delete_safe_audio_button", disabled=not confirm_delete):
                delete_result = delete_dataset_files(safe_files)
                st.success(
                    f"{len(delete_result['deleted'])} legacy audio files were deleted successfully."
                    if get_language() == "en"
                    else f"{len(delete_result['deleted'])} file audio lama berhasil dihapus."
                )
                if delete_result["failed"]:
                    st.warning(
                        f"{len(delete_result['failed'])} files could not be deleted."
                        if get_language() == "en"
                        else f"Ada {len(delete_result['failed'])} file yang gagal dihapus."
                    )
                    with st.expander("Failed deletion details" if get_language() == "en" else "Detail file yang gagal dihapus", expanded=False):
                        for item in delete_result["failed"]:
                            st.markdown(f"- `{item['path']}`")
                            st.caption(f"{'Reason' if get_language() == 'en' else 'Alasan'}: {item['reason']}")
                st.rerun()
        else:
            st.caption("No files are safe to delete yet." if get_language() == "en" else "Belum ada file yang aman dihapus.")

        if unsafe_files:
            with st.expander("Files not safe to delete yet" if get_language() == "en" else "File yang belum aman dihapus", expanded=False):
                preview_unsafe = unsafe_files[:20]
                for path in preview_unsafe:
                    st.markdown(f"- `{path}`")
                if len(unsafe_files) > len(preview_unsafe):
                    st.caption(
                        f"There are still {len(unsafe_files) - len(preview_unsafe)} more files that are not safe to delete."
                        if get_language() == "en"
                        else f"Masih ada {len(unsafe_files) - len(preview_unsafe)} file lain yang belum aman dihapus."
                    )
                if skipped_unsafe_files:
                    st.caption(
                        "Some of the files above were skipped and should be converted first."
                        if get_language() == "en"
                        else "Sebagian file di atas termasuk file skipped yang perlu dikonversi dulu."
                    )

        if latest_sync_report and (latest_sync_report.get("rows_skipped") or []):
            if st.button("Refresh Skipped File Status" if get_language() == "en" else "Refresh Status File Skipped", key="refresh_skipped_status_button"):
                refreshed_report = dict(latest_sync_report)
                refreshed_report["rows_skipped"] = [
                    item for item in (latest_sync_report.get("rows_skipped") or []) if Path(item.get("path", "")).exists()
                ]
                save_sync_report(refreshed_report)
                st.session_state["last_sync_report"] = refreshed_report
                st.success("Skipped file status has been refreshed." if get_language() == "en" else "Status file skipped sudah diperbarui.")
                st.rerun()

        if st.button("Train Model from feature store" if get_language() == "en" else "Train Model dari feature store", key="train_model_button"):
            with st.spinner("Model training is running..." if get_language() == "en" else "Training model sedang berjalan..."):
                command = [
                    sys.executable,
                    "train_model.py",
                    "--dataset",
                    str(DATASET_CSV_PATH),
                    "--feature-store",
                    str(FEATURE_STORE_PATH),
                    "--output",
                    "model.joblib",
                ]
                result = subprocess.run(
                    command,
                    cwd=Path(__file__).parent,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
            if result.returncode == 0:
                st.success("Training is complete. model.joblib has been updated." if get_language() == "en" else "Training selesai. model.joblib sudah diperbarui.")
                if result.stdout.strip():
                    st.code(result.stdout.strip())
            else:
                st.error("Training failed to run." if get_language() == "en" else "Training gagal dijalankan.")
                if result.stdout.strip():
                    st.code(result.stdout.strip())
                if result.stderr.strip():
                    st.code(result.stderr.strip())


def render_how_it_works_panel():
    if DEMO_MODE:
        st.info(t("online_demo_info"))
    st.markdown('<div class="aa-article">', unsafe_allow_html=True)
    is_en = get_language() == "en"
    panel_title = "How This Detector Works" if is_en else "Bagaimana Detector Ini Bekerja"
    panel_subtitle = (
        "An article-style explanation to help users understand how the model works, which features are read, its limitations, and how to interpret the dashboard honestly."
        if is_en else
        "Penjelasan versi artikel untuk memahami cara kerja model, fitur yang dibaca, keterbatasan, dan cara menafsirkan hasil dashboard dengan jujur."
    )
    doc_title = "Audio AI Checker: How It Works" if is_en else "Audio AI Checker: Cara Kerjanya"
    doc_caption = (
        "A dashboard-native documentation page written to feel like a product explainer."
        if is_en else
        "Versi dokumentasi di dalam dashboard, ditulis agar lebih mudah dibaca seperti halaman penjelasan produk."
    )
    callout_title = "Quick Reading Guide" if is_en else "Panduan Baca Cepat"
    callout_text = (
        "Read this page as a guide to understand the dashboard's decisions, not as a promise that the model is always right."
        if is_en else
        "Baca halaman ini sebagai panduan memahami keputusan dashboard, bukan sebagai janji bahwa model selalu benar."
    )
    info_text = (
        "This tool tries to assess whether an audio file is closer to Human, Hybrid, or AI patterns. The output should be read as a screening assistant, not an automatic accusation tool."
        if is_en else
        "Tool ini mencoba menilai apakah audio lebih dekat ke pola Human, Hybrid, atau AI. Hasilnya sebaiknya dibaca sebagai screening assistant, bukan alat tuduhan otomatis."
    )

    summary_cards = [
        (
            "Model Core",
            "Forensic heuristics + calibrated LogisticRegression." if is_en else "Heuristik forensik + LogisticRegression terkalibrasi.",
        ),
        (
            "Decision Style",
            "Conservative. The model may suspect AI, but FAIL still needs strong evidence." if is_en else "Konservatif. Model boleh curiga, tetapi FAIL butuh bukti kuat.",
        ),
        (
            "Training Data",
            "Learns from a feature store so older knowledge is preserved." if is_en else "Belajar dari feature store agar pengetahuan lama tidak hilang.",
        ),
        (
            "Best Use",
            "Early screening, manual review, and dataset growth." if is_en else "Screening awal, audit manual, dan pengayaan dataset.",
        ),
    ]

    st.markdown(
        f"""
        <div class="aa-panel">
            <div class="aa-panel-title">{panel_title}</div>
            <div class="aa-panel-subtitle">{panel_subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"### {doc_title}")
    st.caption(doc_caption)
    st.markdown(
        f"""
        <div class="aa-callout">
            <div class="aa-callout-title">{callout_title}</div>
            <div>{callout_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    card_cols = st.columns(2)
    for idx, (title, body) in enumerate(summary_cards):
        with card_cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(body)
    st.info(info_text)

    st.markdown("### How the Model Works" if is_en else "### Bagaimana Model Bekerja")
    st.write(
        "The engine uses two layers: first, forensic heuristics based on audio features; second, an ML layer that reads the overall feature pattern. The current ML model is a calibrated `LogisticRegression`, not a large neural network."
        if is_en else
        "Engine utama saat ini memakai kombinasi dua lapis: pertama, pembacaan heuristik forensik berbasis fitur audio; kedua, model ML untuk membaca pola fitur secara lebih global. Model ML yang dipakai sekarang adalah `LogisticRegression` yang dikalibrasi probabilitasnya, jadi bukan neural network besar."
    )
    st.write(
        "That is why the dashboard shows both `Forensic decision` for the conservative screening call and `Prediksi ML` for the model tendency."
        if is_en else
        "Karena itu, dashboard menampilkan dua sudut pandang sekaligus: `Forensic decision` untuk keputusan screening yang konservatif, dan `Prediksi ML` untuk melihat kecenderungan model."
    )

    st.markdown("### Training Process" if is_en else "### Proses Training")
    st.write(
        "The model is trained from labeled `human`, `hybrid`, and `ai` audio. Each file is converted into numerical features first, and those features are stored in `training_features.csv` so older learning can survive even if raw audio is cleaned up later."
        if is_en else
        "Model dilatih dari kumpulan audio berlabel `human`, `hybrid`, dan `ai`. Setiap file tidak langsung dipelajari sebagai waveform mentah, tetapi diubah dulu menjadi kumpulan angka fitur. Kumpulan fitur ini disimpan di `training_features.csv`, sehingga pengetahuan numerik lama bisa dipertahankan walaupun audio mentahnya nanti dibersihkan dari folder dataset."
    )
    st.markdown(
        """
        - Audio files are labeled through the dashboard.
        - Numerical features are saved into the feature store.
        - Retraining reads the feature store instead of requiring legacy raw audio.
        - Training outputs are saved into `model.joblib`.
        """
        if is_en else
        """
        - File audio diberi label yang benar melalui dashboard.
        - Fitur numerik disimpan ke feature store.
        - Training ulang model membaca feature store, bukan wajib dari audio lama.
        - Hasil training disimpan ke `model.joblib`.
        """
    )

    st.markdown("### Feature Analysis" if is_en else "### Analisis Fitur")
    st.write(
        "The detector reads several main feature groups:"
        if is_en else
        "Detector ini membaca beberapa kelompok fitur utama:"
    )
    st.markdown(
        """
        1. **Basic Spectral Features**
           Reads spectrum texture, loudness, centroid, flatness, entropy, rolloff, and MFCC.
        2. **Harmonic & Vocal Proxies**
           Reads harmonic stability, phase coherence, pitch transition, and vocal-vs-music proxy ratios.
        3. **Long-Range Pattern Analysis**
           Reads repetition and similarity across 60-second, 120-second, and 180-second windows.
        4. **Fingerprint Signals**
           Looks for generator-like patterns such as texture tiling, air-band stability, and HF shimmer behavior.
        """
        if is_en else
        """
        1. **Basic Spectral Features**
           Melihat tekstur spektrum, loudness, centroid, flatness, entropy, rolloff, dan MFCC.
        2. **Harmonic & Vocal Proxies**
           Membaca kestabilan harmoni, phase coherence, pitch transition, serta proxy rasio vocal-vs-music.
        3. **Long-Range Pattern Analysis**
           Membaca pola repetisi dan kemiripan antar segmen waktu 60 detik, 120 detik, dan 180 detik.
        4. **Fingerprint Signals**
           Mencari pola yang terasa generator-like, misalnya texture tiling, air-band stability, dan HF shimmer behavior.
        """
    )

    st.markdown("### How to Read the Result" if is_en else "### Cara Membaca Hasil")
    st.markdown(
        """
        - **PASS**: no strong evidence is currently serious enough to accuse the file of being AI.
        - **REVIEW**: something still needs manual checking, but strong AI evidence is not complete yet.
        - **FAIL**: the combination of strong evidence is serious enough that the file should be considered suspicious.
        """
        if is_en else
        """
        - **PASS**: belum ada strong evidence yang cukup untuk menuduh AI.
        - **REVIEW**: ada hal yang perlu dicek manual, tapi bukti keras AI belum cukup.
        - **FAIL**: kombinasi bukti kuat sudah cukup serius dan audio patut dicurigai.
        """
    )
    st.write(
        "If `Prediksi ML` leans toward AI while the `Forensic decision` stays at `REVIEW`, it means the model is suspicious but the strong forensic evidence is still incomplete. This is intentional so the dashboard stays honest and not overly aggressive."
        if is_en else
        "Kalau `Prediksi ML` tinggi ke AI tetapi `Forensic decision` masih `REVIEW`, itu berarti model curiga, namun bukti forensik kuatnya belum lengkap. Ini dibuat sengaja supaya dashboard lebih jujur dan tidak terlalu agresif."
    )

    st.markdown("### Limitations and Considerations" if is_en else "### Keterbatasan dan Pertimbangan")
    st.markdown(
        """
        - Very long audio can make repetition patterns look stronger than they really are.
        - Modern production with heavy mastering, strong AutoTune, or very polished textures can look AI-like.
        - Instrumental songs or tracks with limited vocals make some vocal metrics less informative.
        - This model learns from a local dataset that keeps evolving, so label quality matters a lot.
        - Generator archetypes remain probabilistic and often end up as `Unknown generator`.
        """
        if is_en else
        """
        - Durasi audio yang sangat panjang bisa membuat pola repetisi terlihat lebih kuat dari yang sebenarnya.
        - Produksi modern dengan mastering padat, AutoTune berat, atau tekstur yang sangat rapi bisa terbaca mirip AI.
        - Lagu instrumental atau minim vokal membuat beberapa metrik vokal menjadi kurang informatif.
        - Model ini belajar dari dataset lokal yang terus berkembang, jadi kualitas label dataset sangat berpengaruh.
        - Hasil `archetype` generator tetap probabilistik dan sering wajar berakhir di `Unknown generator`.
        """
    )

    st.markdown("### Next Steps" if is_en else "### Langkah Berikutnya")
    st.markdown(
        """
        - Add a cleaner and more balanced dataset.
        - Improve plain-language explanations so non-technical users understand results faster.
        - Add richer training progress visuals and stronger dataset audits.
        - Improve the fingerprint module so weak evidence, strong evidence, and headline verdicts stay more consistent.
        """
        if is_en else
        """
        - Menambah dataset yang lebih seimbang dan lebih bersih.
        - Memperkuat penjelasan awam agar user non-teknis lebih cepat paham.
        - Menambah visual training progress dan audit dataset yang lebih lengkap.
        - Mengembangkan fingerprint module agar semakin konsisten antara weak evidence, strong evidence, dan headline verdict.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_report(report: AnalysisReport, mode: str, y=None, sr=None, history_entry=None):
    st.divider()
    executive_summary_text = translate_generated_text(report.executive_summary)
    simple_explanation_text = translate_generated_text(report.simple_explanation)
    main_issue_text = translate_generated_text(report.main_issue)
    fix_area_text = translate_generated_text(report.fix_area)
    practical_steps = translate_generated_list(report.practical_steps)
    red_flags = translate_generated_list(report.red_flags)
    strong_red_flags = translate_generated_list(report.strong_red_flags)
    weak_indicators = translate_generated_list(report.weak_indicators)
    production_mimic_indicators = translate_generated_list(report.production_mimic_indicators)
    fingerprint_summary_text = translate_generated_text(report.fingerprint_summary)
    fingerprint_signals = translate_generated_list(report.fingerprint_signals)
    limitations_warnings = translate_generated_list(report.limitations_warnings)
    humanization_guide = translate_generated_list(report.humanization_guide)
    verdict_confidence_reason = translate_generated_text(report.verdict_confidence_reason)
    decision_hint_text = translate_generated_text(report.decision_hint)
    ai_verdict_confidence_reason = translate_generated_text(report.ai_verdict_confidence_reason)
    fail_hint_text = translate_generated_text(report.fail_hint)
    fingerprint_confidence_reason = translate_generated_text(report.fingerprint_confidence_reason)
    if history_entry is not None:
        st.info(
            f"{t('rendered_history_for')} `{history_entry['source_name']}` {('from' if get_language() == 'en' else 'dari')} {format_relative_time(history_entry['created_at'])}."
        )
        if history_entry.get("reference_link"):
            st.caption(f"{'Reference link' if get_language() == 'en' else 'Link referensi'}: {history_entry['reference_link']}")

    with st.container(border=True):
        st.markdown(f"### {ui_text(report.overall_verdict)}")
        st.caption(f"{t('forensic_decision')}: {ui_text(report.screening_outcome)}")
        st.caption(f"{t('ml_prediction')}: {ui_text(report.model_label)}")
        st.caption(
            f"{t('headline_probabilities')}: {t('human')} {report.headline_probabilities.human}% | "
            f"{t('hybrid')} {report.headline_probabilities.hybrid}% | "
            f"AI {report.headline_probabilities.ai}%"
        )
        st.caption(f"{t('engine')}: {ui_text(report.analysis_engine)}")
        st.caption(f"{t('decision_confidence')} ({ui_text(report.screening_outcome)}): {ui_text(report.verdict_confidence_label)}")
        if verdict_confidence_reason:
            st.caption(verdict_confidence_reason)
        if decision_hint_text:
            st.caption(decision_hint_text)
        st.caption(f"{t('fail_confidence')}: {ui_text(report.ai_verdict_confidence_label)}")
        if ai_verdict_confidence_reason:
            st.caption(ai_verdict_confidence_reason)
        if fail_hint_text:
            st.caption(fail_hint_text)
        st.markdown(executive_summary_text)
        c1, c2, c3 = st.columns(3)
        c1.metric(t("human"), f"{report.headline_probabilities.human}%")
        c2.metric(t("hybrid"), f"{report.headline_probabilities.hybrid}%")
        c3.metric(t("pure_ai"), f"{report.headline_probabilities.ai}%")
        st.caption(
            f"{t('headline_source_caption')} ({report.headline_probability_source}). "
            f"{t('detail_caption_rest')}"
        )

    # Create tabs for structured view
    tab_forensic, tab_ai_expert, tab_raw_data = st.tabs([t("main_analysis_tab"), t("ai_auditor_tab"), t("raw_data_tab")])

    with tab_forensic:
        with st.container(border=True):
            st.markdown(f"### {t('meaning_of_result')}")
            is_en = get_language() == "en"
            title_explanations = {
                "Likely Human Audio": "This title means the system did not find strong enough evidence to seriously suspect AI." if is_en else "Judul ini berarti sistem tidak menemukan bukti kuat yang cukup untuk mencurigai AI.",
                "Possible AI Detected": "This title means there is a strong enough combination of forensic evidence to suspect AI or heavy synthetic processing." if is_en else "Judul ini berarti ada kombinasi bukti forensik yang cukup kuat sehingga audio patut dicurigai sebagai AI atau proses sintetis berat.",
                "AI Suspected (Review)": "This title means the model is fairly suspicious of AI, but the final decision remains review because strong forensic evidence is still incomplete." if is_en else "Judul ini berarti model cukup curiga ke AI, tetapi keputusan akhirnya tetap review karena bukti forensik kuatnya belum lengkap.",
                "Likely Human (Review)": "This title means the audio leans human, but there is still enough context for the system to choose manual review." if is_en else "Judul ini berarti audio cenderung human, tetapi masih ada konteks yang membuat sistem memilih review manual.",
                "Hybrid / Unclear (Review)": "This title means the result is still mixed or not yet clear enough, so the system chooses manual review." if is_en else "Judul ini berarti hasilnya masih campuran atau belum cukup jelas, jadi sistem memilih review manual.",
            }
            forensic_explanations = {
                "PASS": "This is the safest forensic decision. It means there is no strong enough combination of evidence to accuse AI." if is_en else "Ini keputusan forensik paling aman. Artinya tidak ada kombinasi bukti kuat yang cukup untuk menuduh AI.",
                "REVIEW": "This means the file still needs manual checking. There are suspicious or unstable signs, but hard AI evidence is still insufficient." if is_en else "Ini artinya file perlu dicek manual. Ada beberapa hal yang mencurigakan atau tidak stabil, tetapi bukti keras AI belum cukup.",
                "FAIL": "This means the combination of strong evidence is serious enough that the file should be suspected as AI." if is_en else "Ini artinya ada kombinasi bukti kuat yang cukup serius, jadi file patut dicurigai sebagai AI.",
            }
            model_explanations = {
                "Human": "The ML model leans most strongly toward reading this audio as human." if is_en else "Model ML paling condong membaca audio ini sebagai human.",
                "Hybrid": "The ML model leans most strongly toward reading this audio as mixed or in the gray area between human and AI-like patterns." if is_en else "Model ML paling condong membaca audio ini sebagai campuran atau berada di area abu-abu antara human dan AI-like pattern.",
                "AI": "The ML model leans most strongly toward reading this audio as AI." if is_en else "Model ML paling condong membaca audio ini sebagai AI.",
            }
            fingerprint_explanations = {
                "Low": "The generator-like fingerprint is not very strong." if is_en else "Pola fingerprint generator tidak terlalu kuat.",
                "Medium": "Some patterns resemble generator audio, but they can still overlap with modern production choices." if is_en else "Ada beberapa pola yang mirip audio generator, tetapi masih bisa tercampur dengan produksi modern.",
                "High": "There are quite a few technical patterns that resemble generator fingerprints or extremely polished audio processing." if is_en else "Ada cukup banyak pola teknis yang mirip fingerprint audio generator atau audio yang diproses sangat rapi.",
            }

            st.markdown(f"- **{'Title' if is_en else 'Judul'}:** {title_explanations.get(report.overall_verdict, 'This title is a short summary of the final system reading.' if is_en else 'Judul ini adalah ringkasan singkat dari pembacaan akhir sistem.')}")
            st.markdown(f"- **{t('forensic_decision')}:** {forensic_explanations.get(report.screening_outcome, 'This is the main forensic decision of the system.' if is_en else 'Ini adalah keputusan forensik utama sistem.')}")
            st.markdown(f"- **{t('ml_prediction')}:** {model_explanations.get(report.model_label, 'This is the class most favored by the ML model.' if is_en else 'Ini adalah kelas yang paling dipilih oleh model ML.')}")
            st.markdown(
                f"- **{t('headline_probabilities')}:** "
                + (
                    f"This shows the score split used for the title and main decision: Human {report.headline_probabilities.human}%, Hybrid {report.headline_probabilities.hybrid}%, AI {report.headline_probabilities.ai}%. In this case the headline numbers use `{report.headline_probability_source}` as the basis. The details below still show ML probabilities and per-domain heuristics."
                    if is_en
                    else f"Ini menunjukkan pembagian skor yang dipakai untuk judul dan keputusan utama: Human {report.headline_probabilities.human}%, Hybrid {report.headline_probabilities.hybrid}%, AI {report.headline_probabilities.ai}%. Di kasus ini angka headline memakai basis `{report.headline_probability_source}`. Detail di bawah tetap menampilkan probabilitas ML dan heuristik per domain."
                )
            )
            st.markdown(
                f"- **{t('decision_confidence')}:** "
                + (
                    f"This shows how confident the system is about the `{report.screening_outcome}` decision. It does not mean the system is certain the file is AI."
                    if is_en
                    else f"Ini menunjukkan seberapa yakin sistem terhadap keputusan `{report.screening_outcome}`. Ini bukan berarti sistem yakin file ini pasti AI."
                )
            )
            st.markdown(
                f"- **{t('fail_confidence')}:** "
                + (
                    "This shows whether the available evidence is already strong enough to escalate to `FAIL`."
                    if is_en
                    else "Ini menunjukkan apakah bukti yang ada sudah cukup kuat untuk naik ke keputusan `FAIL`."
                )
            )
            st.markdown(f"- **{'Fingerprint' if is_en else 'Fingerprint'}:** {fingerprint_explanations.get(report.fingerprint_level, 'This shows how strong the visible generator-like pattern is.' if is_en else 'Ini menunjukkan seberapa kuat pola generator-like yang terlihat.')}")
            st.markdown(
                f"- **{'Archetype' if is_en else 'Arketipe'}:** "
                + (
                    f"If a generator-like pattern exists, the system judges this file to be closest to `{report.fingerprint_archetype}`. If it says `Unknown generator`, the pattern is not distinctive enough to point to a specific generator type."
                    if is_en
                    else f"Jika ada pola generator-like, sistem menilai file ini paling dekat ke `{report.fingerprint_archetype}`. Kalau tertulis `Unknown generator`, artinya pola itu belum cukup khas untuk diarahkan ke tipe generator tertentu."
                )
            )

        with st.container(border=True):
            st.markdown(f"### {t('evidence_card')}")
            indicator_labels = {
                "S1a_texture_tiling": "S1a Texture tiling (mel)",
                "S1b_structural_tiling": "S1b Structural tiling (chroma/mid)",
                "S1_tiling_kuat": "S1 Tiling kuat",
                "S2_weak_air_stability": "S2 weak Air stability",
                "S2_air_band_stasioner": "S2 HF shimmer + air band mencurigakan",
                "S3_vocal_lock_proxy": "S3 Vocal synthesis proxy",
                "S4_metadata_ai": "S4 Metadata AI eksplisit",
            }
            for key, active in report.strong_indicator_status.items():
                status = ("Yes" if active else "No") if get_language() == "en" else ("Ya" if active else "Tidak")
                st.markdown(f"- **{indicator_labels.get(key, key)}:** {status}")

            if strong_red_flags:
                st.markdown("**Strong indicators:**" if get_language() == "en" else "**Indikator kuat:**")
                for item in strong_red_flags:
                    st.error(item)
            if weak_indicators:
                st.markdown("**Weak indicators:**" if get_language() == "en" else "**Indikator lemah:**")
                for item in weak_indicators:
                    st.warning(item)
            if production_mimic_indicators:
                st.markdown("**Production-mimic indicators:**" if get_language() == "en" else "**Indikator mirip produksi modern:**")
                for item in production_mimic_indicators:
                    st.info(item)

        with st.container(border=True):
            st.markdown(f"### {t('fingerprint_module')}")
            st.markdown(f"**{t('generator_fingerprint')}:** {ui_text(report.fingerprint_level)} ({report.fingerprint_score}/100)")
            st.markdown(f"**{t('most_consistent_with')}:** {ui_text(report.fingerprint_archetype)}")
            st.markdown(f"**{t('fingerprint_confidence')}:** {ui_text(report.fingerprint_confidence_label)}")
            if fingerprint_confidence_reason:
                st.caption(fingerprint_confidence_reason)
            st.write(fingerprint_summary_text)
            if fingerprint_signals:
                for item in fingerprint_signals:
                    st.markdown(f"- {item}")
            if report.fingerprint_score_components:
                st.markdown(f"**{t('score_breakdown')}:**")
                for key, value in report.fingerprint_score_components.items():
                    st.markdown(f"- `{key}`: `{value}`")
            if report.fingerprint_metrics:
                st.markdown("**Fingerprint trigger metrics:**" if get_language() == "en" else "**Metrik pemicu fingerprint:**")
                for key, value in report.fingerprint_metrics.items():
                    st.markdown(f"- `{key}`: `{value}`")

        with st.container(border=True):
            st.markdown("### Plain-language Explanation" if get_language() == "en" else "### Bahasa Sederhana")
            st.write(simple_explanation_text)
            st.markdown(f"**{'Main issue' if get_language() == 'en' else 'Masalah utama'}:** {main_issue_text}")
            st.markdown(f"**{'Fix area' if get_language() == 'en' else 'Perbaiki di bagian'}:** {fix_area_text}")
            st.markdown("**Practical steps:**" if get_language() == "en" else "**Langkah praktis:**")
            for step in practical_steps:
                st.markdown(f"- {step}")

    with tab_ai_expert:
        with st.container(border=True):
            st.markdown("### AI Expert Insight (Thinking Logic)" if get_language() == "en" else "### Insight AI Expert (Logika Berpikir)")
            if report.expert_insight:
                # Handle potential error response from API
                if "error" in report.expert_insight:
                    st.error(report.expert_insight["error"])
                else:
                    # Executive Summary
                    st.markdown(f"**{'Executive Summary' if get_language() == 'en' else 'Ringkasan Eksekutif'}:**\n\n{report.expert_insight.get('executive_summary', '')}")
                    
                    # Findings & Production Mimics
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        st.markdown("**Top Findings:**" if get_language() == "en" else "**Temuan Utama:**")
                        for finding in report.expert_insight.get("top_findings", []):
                            st.markdown(f"- {finding}")
                    with col_f2:
                        st.markdown("**Could be Production (False Positives):**" if get_language() == "en" else "**Mungkin Hanya Produksi (False Positive):**")
                        for mimic in report.expert_insight.get("could_be_production", []):
                            st.markdown(f"- {mimic}")
                    
                    # Manual Checks
                    st.markdown("**Manual Checks Needed:**" if get_language() == "en" else "**Pemeriksaan Manual yang Disarankan:**")
                    for check in report.expert_insight.get("manual_checks", []):
                        st.markdown(f"- [ ] {check}")
                    
                    # Confidence Explainer
                    with st.expander("AI Reasoning & Gaps (Deep Logic)" if get_language() == "en" else "Logika AI & Celah Analisis", expanded=False):
                        explainer = report.expert_insight.get("confidence_explainer", {})
                        st.markdown(f"**{'Reasoning' if get_language() == 'en' else 'Penalaran'}:** {explainer.get('reasoning', '')}")
                        st.markdown(f"**{'Gap Analysis' if get_language() == 'en' else 'Analisis Celah'}:** {explainer.get('gap', '')}")
            else:
                st.caption("Click the button below to ask the AI module (OpenAI) for a deeper forensic explanation." if get_language() == "en" else "Klik tombol di bawah untuk meminta modul AI (OpenAI) menganalisa data forensik ini lebih mendalam.")
                if st.button("Generate Expert Insight" if get_language() == "en" else "Buat Insight Expert", key=f"gen_insight_{report.metadata.filename}"):
                    with st.spinner("Contacting AI Expert..." if get_language() == "en" else "Menghubungi AI Expert..."):
                        selected_history_id = history_entry.get("id") if history_entry else None
                        insight_json = generate_expert_insight(report)
                        if insight_json:
                            report.expert_insight = insight_json
                            # Persistence: Update history if applicable
                            if selected_history_id:
                                history_items = load_history()
                                for entry in history_items:
                                    if entry.get("id") == selected_history_id:
                                        entry["report"]["expert_insight"] = insight_json
                                        break
                                save_history(history_items)
                            st.rerun()

    with tab_raw_data:
        st.markdown("### Raw Evidence Data (JSON)" if get_language() == "en" else "### Data Bukti Mentah (JSON)")
        st.markdown("**1. Copy System Prompt (AI Instruction):**" if get_language() == "en" else "**1. Copy System Prompt (Instruksi AI):**")
        st.code(
            "You are a senior audio forensics auditor. You do not hear the audio; you only read metrics. Explain honestly. Do not mention Suno/Udio as certainty unless there is explicit metadata or a very strong fingerprint."
            if get_language() == "en"
            else "Anda auditor forensik audio senior. Anda tidak mendengar audio, hanya membaca metrik. Jelaskan secara jujur. Jangan menyebut Suno/Udio sebagai kepastian kecuali ada metadata eksplisit atau fingerprint sangat kuat.",
            language="text",
        )
        
        st.markdown("**2. Copy Data JSON (Technical Metrics):**" if get_language() == "en" else "**2. Copy Data JSON (Metrik Teknis):**")
        st.caption("Copy the JSON below and paste it into ChatGPT or another external analysis tool." if get_language() == "en" else "Copy JSON di bawah ini untuk ditempelkan ke ChatGPT atau alat analisa eksternal lainnya.")
        payload = get_minimal_payload(report)
        st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")
        st.info("The data above is the technical fingerprint sent to the AI module. It does not contain raw audio." if get_language() == "en" else "Data di atas adalah 'sidik jari' teknis lagu ini yang dikirim ke modul AI. Tidak mengandung audio mentah.")

    if history_entry is None and not DEMO_MODE:
        latest_upload = st.session_state.get("latest_analyzed_upload")
        if latest_upload:
            with st.container(border=True):
                st.markdown("### Save to Dataset" if get_language() == "en" else "### Simpan ke Dataset")
                st.caption("Choose the correct label if this uploaded file should become additional training data." if get_language() == "en" else "Pilih label yang benar jika file upload ini ingin dijadikan tambahan dataset training.")
                c1, c2, c3 = st.columns(3)
                if c1.button("Add to Human Dataset" if get_language() == "en" else "Tambahkan ke Dataset Human", key="save_dataset_human"):
                    saved_path = add_uploaded_audio_to_dataset("human", latest_upload["name"], latest_upload["bytes"])
                    save_training_features("human", report, saved_path)
                    st.success(f"File saved to human dataset: {saved_path.name}" if get_language() == "en" else f"File disimpan ke dataset human: {saved_path.name}")
                if c2.button("Add to Hybrid Dataset" if get_language() == "en" else "Tambahkan ke Dataset Hybrid", key="save_dataset_hybrid"):
                    saved_path = add_uploaded_audio_to_dataset("hybrid", latest_upload["name"], latest_upload["bytes"])
                    save_training_features("hybrid", report, saved_path)
                    st.success(f"File saved to hybrid dataset: {saved_path.name}" if get_language() == "en" else f"File disimpan ke dataset hybrid: {saved_path.name}")
                if c3.button("Add to AI Dataset" if get_language() == "en" else "Tambahkan ke Dataset AI", key="save_dataset_ai"):
                    saved_path = add_uploaded_audio_to_dataset("ai", latest_upload["name"], latest_upload["bytes"])
                    save_training_features("ai", report, saved_path)
                    st.success(f"File saved to AI dataset: {saved_path.name}" if get_language() == "en" else f"File disimpan ke dataset ai: {saved_path.name}")
                st.caption("After a few files have been added, you can retrain the model from the updated dataset." if get_language() == "en" else "Setelah beberapa file ditambahkan, kamu bisa train ulang model dari dataset yang sudah diperbarui.")

    st.markdown(f"### {t('detail')}")
    st.markdown(f"- {t('duration')}: {int(report.metadata.duration_sec)} {'seconds' if get_language() == 'en' else 'detik'}")
    st.markdown(f"- **{t('headline_basis')}:**")
    st.markdown(f"  - {t('source_used')}: {ui_text(report.headline_probability_source)}")
    st.markdown(f"  - {translate_generated_text(report.headline_probabilities.verdict_text)}")
    prob_word = "probability" if get_language() == "en" else "probabilitas"
    st.markdown(f"  - {t('human')}: {prob_word} ({report.headline_probabilities.human}%)")
    st.markdown(f"  - {t('pure_ai')}: {prob_word} ({report.headline_probabilities.ai}%)")
    st.markdown(f"  - {t('hybrid')}: {prob_word} ({report.headline_probabilities.hybrid}%)")

    st.markdown(f"- **{t('ml_probs_detail')}:**")
    st.markdown(f"  - {t('engine_used')}: {ui_text(report.analysis_engine)}")
    st.markdown(f"  - {translate_generated_text(report.final_probabilities.verdict_text)}")
    st.markdown(f"  - {t('human')}: {prob_word} ({report.final_probabilities.human}%)")
    st.markdown(f"  - {t('pure_ai')}: {prob_word} ({report.final_probabilities.ai}%)")
    st.markdown(f"  - {t('hybrid')}: {prob_word} ({report.final_probabilities.hybrid}%)")

    st.markdown(f"- **{t('heuristic_combined')}:**")
    st.markdown(f"  - {translate_generated_text(report.heuristic_combined_analysis.verdict_text)}")
    st.markdown(f"  - {t('human')}: {prob_word} ({report.heuristic_combined_analysis.human}%)")
    st.markdown(f"  - {t('pure_ai')}: {prob_word} ({report.heuristic_combined_analysis.ai}%)")
    st.markdown(f"  - {t('hybrid')}: {prob_word} ({report.heuristic_combined_analysis.hybrid}%)")

    st.markdown(f"- **{t('heuristic_spectral')}:**")
    st.markdown(f"  - {translate_generated_text(report.spectral_analysis.verdict_text)}")
    st.markdown(f"  - {t('human')}: {prob_word} ({report.spectral_analysis.human}%)")
    st.markdown(f"  - {t('pure_ai')}: {prob_word} ({report.spectral_analysis.ai}%)")
    st.markdown(f"  - {t('hybrid')}: {prob_word} ({report.spectral_analysis.hybrid}%)")

    st.markdown(f"- **{t('heuristic_temporal')}:**")
    st.markdown(f"  - {translate_generated_text(report.temporal_analysis.verdict_text)}")
    st.markdown(f"  - {t('human')}: {prob_word} ({report.temporal_analysis.human}%)")
    st.markdown(f"  - {t('pure_ai')}: {prob_word} ({report.temporal_analysis.ai}%)")
    st.markdown(f"  - {t('hybrid')}: {prob_word} ({report.temporal_analysis.hybrid}%)")

    if len(limitations_warnings) > 0:
        st.markdown("---")
        st.markdown("**Warnings & Recommendations:**" if get_language() == "en" else "**Peringatan & Rekomendasi:**")
        for warning in limitations_warnings:
            st.info(warning)
    if not MODEL_IS_RELIABLE:
        st.warning(MODEL_STATUS)
    elif report.confidence_label == "low":
        st.warning("This audio prediction is still low confidence, so the final result falls back to heuristics." if get_language() == "en" else "Prediksi model untuk audio ini masih low confidence, jadi hasil akhir memakai heuristik.")

    if mode == "A":
        st.header("Mode A: Deep Analysis" if get_language() == "en" else "Mode A: Analisis Ultra-Akurat")
        tab1, tab2, tab3, tab4 = st.tabs(
            ["1. Metadata Layer", "2. Detailed DSP Metrics", "3. Advanced Feature Parity", "4. Charts"]
            if get_language() == "en"
            else ["1. Layer Metadata", "2. Metrik DSP Rinci", "3. Advanced Feature Parity", "4. Grafik"]
        )

        with tab1:
            st.json(report.metadata.model_dump())
            if history_entry is not None and history_entry.get("reference_link"):
                st.caption(f"{'Reference link' if get_language() == 'en' else 'Link referensi'}: {history_entry['reference_link']}")
            if report.metadata.ai_tags_found:
                st.error(f"{'AI tags detected in metadata' if get_language() == 'en' else 'AI tags terdeteksi di metadata'}: {report.metadata.ai_tags_found}")
        with tab2:
            st.json(report.dsp.model_dump())
            st.markdown("### Red Flags" if get_language() == "en" else "### Tanda Peringatan")
            for item in red_flags:
                st.warning(item)
            if strong_red_flags:
                st.markdown("### Strong AI Indicators" if get_language() == "en" else "### Indikator AI Kuat")
                for item in strong_red_flags:
                    st.error(item)
            if weak_indicators:
                st.markdown("### Weak Indicators" if get_language() == "en" else "### Indikator Lemah")
                for item in weak_indicators:
                    st.warning(item)
            if production_mimic_indicators:
                st.markdown("### Production-Mimic Indicators" if get_language() == "en" else "### Indikator yang Bisa Mirip Produksi Modern")
                for item in production_mimic_indicators:
                    st.info(item)
            if report.ambiguous_red_flags:
                st.markdown("### Ambiguous Indicators" if get_language() == "en" else "### Indikator Ambigu")
                for item in translate_generated_list(report.ambiguous_red_flags):
                    st.warning(item)
            if report.normal_production_flags:
                st.markdown("### Normal Production Choices That Can Mimic AI" if get_language() == "en" else "### Pilihan Produksi Normal yang Bisa Terbaca Mirip AI")
                for item in translate_generated_list(report.normal_production_flags):
                    st.info(item)
        with tab3:
            st.markdown("### SubmitHub-style Matrix Parity" if get_language() == "en" else "### Matrix Parity Berdasarkan SubmitHub")

            c_w60 = report.submithub.harmonic_consistency_chroma.w60s
            m_w60 = report.submithub.texture_consistency_mel.w60s

            if c_w60 is not None and m_w60 is not None:
                mid_w60 = report.submithub.mid_band_w60s
                air_w60 = report.submithub.air_band_w60s
                d_mid = (
                    f"\n\n**Mid-texture [200-2k] (W60s):**\n- median_sim: {mid_w60.median:.6f}\n- median_sim_jittered: {mid_w60.median_jitter:.6f}\n- drop: {mid_w60.drop_vs_jitter:.6f}"
                    if mid_w60
                    else ""
                )
                mid_w120 = report.submithub.mid_band_w120s
                d_mid_120 = (
                    f"\n\n**Mid-texture [200-2k] (W120s):**\n- median_sim: {mid_w120.median:.6f}\n- median_sim_jittered: {mid_w120.median_jitter:.6f}\n- drop: {mid_w120.drop_vs_jitter:.6f}"
                    if mid_w120
                    else ""
                )
                d_air = (
                    f"\n\n**Air-texture [12k-18k] (W60s):**\n- median_sim: {air_w60.median:.6f}\n- median_sim_jittered: {air_w60.median_jitter:.6f}\n- drop: {air_w60.drop_vs_jitter:.6f}"
                    if air_w60
                    else ""
                )

                st.info(
                    f"**Mel-texture (pooled 0.5s):**\n"
                    f"- median_sim(dt=60s): {m_w60.median:.6f}\n"
                    f"- median_sim_jittered: {m_w60.median_jitter:.6f}\n"
                    f"- drop_60: {m_w60.drop_vs_jitter:.6f}\n\n"
                    f"**Chroma harmonic:**\n"
                    f"- median_sim(dt=60s): {c_w60.median:.6f}\n"
                    f"- median_sim_jittered: {c_w60.median_jitter:.6f}\n"
                    f"- drop_chroma_60: {c_w60.drop_vs_jitter:.6f}"
                    + d_mid
                    + d_mid_120
                    + d_air
                )

            st.json(report.submithub.model_dump())
        with tab4:
            if y is not None and sr is not None:
                st.plotly_chart(plot_waveform_rms(y, sr), use_container_width=True)
                st.plotly_chart(plot_f0(y, sr), use_container_width=True)
                st.plotly_chart(plot_spectrum(y, sr), use_container_width=True)
            else:
                st.info("Charts are not available for history items because the original audio source is not stored." if get_language() == "en" else "Grafik tidak tersedia untuk item history karena source audio asli tidak disimpan.")

    elif mode == "C":
        st.header(t("mode_c"))
        st.markdown("Step-by-step guidance for making edits in a DAW, especially BandLab:" if get_language() == "en" else "Panduan langkah demi langkah untuk melakukan perbaikan di DAW, khususnya BandLab:")
        if len(humanization_guide) > 0:
            for idx, step in enumerate(humanization_guide, start=1):
                st.info(f"{idx}. {step}")
        else:
            st.success("The audio already sounds natural. Not much humanization is needed." if get_language() == "en" else "Audio terlihat sudah natural. Tidak banyak humanisasi diperlukan.")

if DEMO_MODE:
    workspace_tab, how_it_works_tab = st.tabs([t("workspace_tab"), t("how_it_works_tab")])
    with workspace_tab:
        st.caption(t("workspace_public_caption"))
        st.info(t("workspace_public_info"))
    with how_it_works_tab:
        render_how_it_works_panel()
else:
    workspace_tab, training_tab, history_tab, how_it_works_tab = st.tabs(
        [t("workspace_tab"), t("training_tab"), t("history_tab"), t("how_it_works_tab")]
    )
    with workspace_tab:
        st.caption(t("workspace_local_caption"))
    with training_tab:
        render_training_panel()
    with history_tab:
        render_history_panel()
    with how_it_works_tab:
        render_how_it_works_panel()

current_report = None
current_y = None
current_sr = None
current_history_entry = None

if analyze_clicked and uploaded_file is not None:
    upload_bytes = uploaded_file.getbuffer().tobytes()
    progress_container = st.empty()
    detail_placeholder = st.empty()

    def update_analysis_progress(percent: int, message: str):
        render_analysis_progress(progress_container, percent=percent, step_label=message, filename=uploaded_file.name)
        detail_placeholder.caption(message)

    update_analysis_progress(5, "Menyiapkan file upload")
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(upload_bytes)
        tmp_path = tmp_file.name

    try:
        update_analysis_progress(20, "Membaca metadata audio")
        metadata = extract_metadata(tmp_path)

        update_analysis_progress(40, "Memuat waveform audio")
        y, sr = librosa.load(tmp_path, sr=None)

        update_analysis_progress(65, "Menghitung fitur DSP dan forensik")
        dsp, submithub = compute_dsp_metrics(y, sr)
        feature_vector = build_feature_vector(metadata, dsp, submithub)

        update_analysis_progress(80, "Menjalankan prediksi model")
        raw_model_probabilities = predict_probabilities(MODEL_BUNDLE, feature_vector) if MODEL_BUNDLE is not None else None
        prediction_confidence, prediction_reason = assess_prediction_confidence(raw_model_probabilities)
        use_model_output = MODEL_IS_RELIABLE and prediction_confidence != "low"
        model_probabilities = raw_model_probabilities if use_model_output else None
        analysis_engine = "ml-model + heuristics" if use_model_output else "heuristics only"
        confidence_label = prediction_confidence if MODEL_IS_RELIABLE else "low"
        confidence_reason = prediction_reason if MODEL_IS_RELIABLE else MODEL_STATUS

        update_analysis_progress(92, "Menyusun hasil akhir analisis")
        report = evaluate_audio(
            metadata,
            dsp,
            submithub,
            feature_vector=feature_vector,
            ml_probabilities=model_probabilities,
            analysis_engine=analysis_engine,
            confidence_label=confidence_label,
            confidence_reason=confidence_reason,
        )
        entry_id = add_history_entry(
            report=report,
            source_name=uploaded_file.name,
            source_type="Upload",
            reference_link="",
        )
        st.session_state["selected_history_id"] = entry_id
        st.session_state["latest_analyzed_upload"] = {"name": uploaded_file.name, "bytes": upload_bytes}
        current_report = report
        current_y = y
        current_sr = sr
        update_analysis_progress(100, "Analisis selesai")
    finally:
        os.remove(tmp_path)

selected_history_id = st.session_state.get("selected_history_id")
if current_report is None and selected_history_id:
    for entry in load_history():
        if entry["id"] == selected_history_id:
            current_history_entry = entry
            current_report = AnalysisReport.model_validate(entry["report"])
            break

if current_report is not None:
    render_report(current_report, mode, y=current_y, sr=current_sr, history_entry=current_history_entry)



