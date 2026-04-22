import json
import os
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


DEMO_MODE = is_demo_mode()
MODEL_BUNDLE = load_model_bundle()
MODEL_IS_RELIABLE, MODEL_STATUS = assess_model_reliability(MODEL_BUNDLE)
HISTORY_PATH = Path(__file__).with_name("analysis_history.json")
DATASET_ROOT = Path(__file__).with_name("data")
DATASET_CSV_PATH = Path(__file__).with_name("dataset.csv")
FEATURE_STORE_PATH = Path(__file__).with_name(str(DEFAULT_FEATURE_STORE_PATH))
SYNC_REPORT_PATH = Path(__file__).with_name("feature_store_sync_report.json")


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
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(26, 93, 73, 0.12), transparent 26%),
                radial-gradient(circle at top right, rgba(177, 111, 55, 0.10), transparent 22%),
                linear-gradient(180deg, #f7f6f1 0%, #fbfaf7 100%);
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
    st.markdown(
        """
        <div class="aa-hero">
            <div class="aa-kicker">Audio Screening Workspace</div>
            <div class="aa-title">Audio AI Forensics Analyzer</div>
            <div class="aa-subtitle">
                Dashboard ini dipakai untuk analisis audio, evaluasi hasil forensik, penyimpanan data training,
                dan retraining model dari feature store tanpa harus menyimpan semua file audio lama.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_overview():
    dataset_summary = get_dataset_summary()
    history_count = len(load_history())
    reliability_badge = "good" if MODEL_IS_RELIABLE else "warn"
    reliability_text = "Siap dipakai" if MODEL_IS_RELIABLE else "Perlu perhatian"

    st.markdown(
        f"""
        <div class="aa-panel">
            <div class="aa-panel-title">Ringkasan Workspace</div>
            <div class="aa-panel-subtitle">Status model, jumlah dataset, feature store, dan history analisis saat ini.</div>
            <div style="margin-top:0.8rem;">
                <span class="aa-badge {reliability_badge}">Model: {reliability_text}</span>
                <span class="aa-badge soft">Feature store: {get_feature_store_count()} baris</span>
                <span class="aa-badge warn">History: {history_count} item</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Dataset", str(dataset_summary["total"]), "Jumlah file audio yang masih ada di folder data.")
    with col2:
        render_metric_card("Human", str(dataset_summary["human"]), "Sampel berlabel human yang masih tersimpan.")
    with col3:
        render_metric_card("Hybrid", str(dataset_summary["hybrid"]), "Sampel berlabel hybrid yang masih tersimpan.")
    with col4:
        render_metric_card("AI", str(dataset_summary["ai"]), "Sampel berlabel AI yang masih tersimpan.")


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
    if seconds < 60:
        return "baru saja"
    if seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} menit lalu"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours} jam lalu"
    days = seconds // 86400
    return f"{days} hari lalu"


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
        col2.metric("Berhasil", str(added))
        col3.metric("Skipped", str(skipped))
        if current_file:
            st.caption(f"{title}: `{current_file}`")


def render_analysis_progress(container, percent: int, step_label: str, filename: str):
    with container.container():
        st.markdown(f"### {percent}%")
        st.progress(min(max(percent, 0), 100))
        col1, col2 = st.columns(2)
        col1.metric("Progress", f"{percent}%")
        col2.metric("Tahap", step_label)
        st.caption(f"File yang diproses: `{filename}`")


def render_skipped_files(report: Optional[dict]):
    if not report:
        return
    skipped_rows = report.get("rows_skipped") or []
    existing_skipped_rows = [item for item in skipped_rows if Path(item.get("path", "")).exists()]
    skipped_rows = existing_skipped_rows
    if not skipped_rows:
        return

    st.warning(f"Ada {len(skipped_rows)} file yang dilewati. File ini sebaiknya dikonversi ke WAV lalu disinkron ulang.")
    with st.expander("Daftar file yang perlu dikonversi", expanded=True):
        for item in skipped_rows:
            st.markdown(f"- `{item.get('path', '-')}`")
            st.caption(f"Alasan: {normalize_skip_reason(item.get('reason', ''))}")


render_dashboard_header()
render_workspace_overview()

control_col, ops_col = st.columns([1.35, 0.85], gap="large")
with control_col:
    st.markdown(
        """
        <div class="aa-panel">
            <div class="aa-panel-title">Analisis Audio</div>
            <div class="aa-panel-subtitle">Upload file, pilih mode, lalu jalankan analisis dari satu tempat.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload Audio File (WAV/MP3/FLAC)", type=["wav", "mp3", "flac", "ogg"])
    mode = st.radio(
        "Pilih Mode Analisis:",
        [
            "Mode A: Analisis Ultra-Akurat + Grafik",
            "Mode B: Analisis Ringkas (Format SubmitHub)",
            "Mode C: Humanization Editing Focus",
        ],
    )
    analyze_clicked = st.button("Jalankan Analisis", disabled=uploaded_file is None, use_container_width=True)

with ops_col:
    if DEMO_MODE:
        st.markdown(
            """
            <div class="aa-panel">
                <div class="aa-panel-title">Alur Cepat Demo Online</div>
                <div class="aa-panel-subtitle">Mode publik hanya untuk analisis audio. Fitur training dan dataset disembunyikan.</div>
                <ol class="aa-steps">
                    <li>Upload audio dari browser.</li>
                    <li>Pilih mode analisis yang ingin dipakai.</li>
                    <li>Klik <b>Jalankan Analisis</b> lalu baca hasil utamanya.</li>
                    <li>Gunakan hasil sebagai screening assistant, bukan keputusan final otomatis.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="aa-panel">
                <div class="aa-panel-title">Alur Cepat</div>
                <div class="aa-panel-subtitle">Urutan paling aman untuk kerja harian di dashboard ini.</div>
                <ol class="aa-steps">
                    <li>Upload audio lalu jalankan analisis.</li>
                    <li>Jika label file sudah yakin, simpan ke dataset yang benar.</li>
                    <li>Sinkronkan feature store untuk mengamankan pengetahuan numerik.</li>
                    <li>Train model dari feature store.</li>
                    <li>Hapus audio lama yang sudah aman jika ingin merapikan folder.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_history_panel():
    history_items = load_history()
    with st.expander("History Analisis", expanded=False):
        if not history_items:
            st.caption("Belum ada history analisis.")
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
                if st.button("Buka", key=f"open_{entry['id']}"):
                    st.session_state["selected_history_id"] = entry["id"]
                if st.button("Hapus", key=f"delete_{entry['id']}"):
                    remaining = [item for item in history_items if item["id"] != entry["id"]]
                    save_history(remaining)
                    if st.session_state.get("selected_history_id") == entry["id"]:
                        st.session_state["selected_history_id"] = None
                    st.rerun()


def render_training_panel():
    with st.expander("Train Model", expanded=False):
        st.caption("Jalankan training ulang dari training_features.csv. Audio lama boleh dihapus setelah fiturnya tersimpan.")
        st.caption(f"Feature store: {FEATURE_STORE_PATH.name}")
        st.caption(f"Jumlah baris feature store saat ini: {get_feature_store_count()}")
        top_actions_col1, top_actions_col2 = st.columns([1, 3])
        with top_actions_col1:
            if st.button("Refresh Dataset Index", key="refresh_dataset_index_button"):
                rebuild_dataset_csv()
                st.success("dataset.csv berhasil disinkronkan ulang dari isi folder data.")
                st.rerun()
        st.markdown("**Langkah penggunaan:**")
        st.markdown("1. Jika kamu masih punya audio lama di folder `data`, klik `Sinkronkan Feature Store dari Dataset Lama` sekali dulu.")
        st.markdown("2. Setelah sinkron selesai, feature store akan menyimpan pengetahuan numerik dari file lama.")
        st.markdown("3. Untuk file baru, analisis dulu lalu klik `Tambahkan ke Dataset Human/Hybrid/AI`.")
        st.markdown("4. Setelah beberapa data baru masuk, klik `Train Model dari feature store`.")
        st.markdown("5. Setelah training selesai dan feature store aman, audio lama boleh dihapus kalau memang tidak ingin disimpan.")
        if st.button("Sinkronkan Feature Store dari Dataset Lama", key="sync_feature_store_button"):
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
                    title="Sedang diproses",
                )
                if payload.get("status") == "skipped":
                    status_placeholder.caption(
                        f"File dilewati: `{current_file}`. Alasan: {normalize_skip_reason(payload.get('reason', ''))}"
                    )

            result = build_feature_store_from_dataset(
                dataset_csv=str(DATASET_CSV_PATH),
                feature_store_csv=str(FEATURE_STORE_PATH),
                progress_callback=on_progress,
            )
            save_sync_report(result)
            st.session_state["last_sync_report"] = result
            st.success(
                f"Sinkronisasi selesai. Ditambahkan {result['rows_added']} baris fitur dari {result['rows_seen']} file dataset."
            )
            render_skipped_files(result)
        latest_sync_report = st.session_state.get("last_sync_report") or load_sync_report()
        render_skipped_files(latest_sync_report)

        safe_files, unsafe_files = get_safe_delete_candidates()
        skipped_paths = {item.get("path", "") for item in (latest_sync_report or {}).get("rows_skipped", [])}
        skipped_unsafe_files = [path for path in unsafe_files if str(path) in skipped_paths]

        st.markdown("**Pembersihan Audio Lama**")
        st.caption("Hanya file yang sudah tercatat di feature store yang aman dihapus. File skipped tidak akan ikut dihapus.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Aman dihapus", str(len(safe_files)))
        col2.metric("Belum aman", str(len(unsafe_files)))
        col3.metric("Skipped", str(len(skipped_unsafe_files)))

        if safe_files:
            with st.expander("Lihat daftar file yang aman dihapus", expanded=False):
                preview_safe = safe_files[:20]
                for path in preview_safe:
                    st.markdown(f"- `{path}`")
                if len(safe_files) > len(preview_safe):
                    st.caption(f"Masih ada {len(safe_files) - len(preview_safe)} file lain yang juga aman dihapus.")

            confirm_delete = st.checkbox(
                "Saya yakin ingin menghapus file audio lama yang sudah aman di feature store",
                key="confirm_safe_delete",
            )
            if st.button("Hapus Audio Lama yang Sudah Aman", key="delete_safe_audio_button", disabled=not confirm_delete):
                delete_result = delete_dataset_files(safe_files)
                st.success(f"{len(delete_result['deleted'])} file audio lama berhasil dihapus.")
                if delete_result["failed"]:
                    st.warning(f"Ada {len(delete_result['failed'])} file yang gagal dihapus.")
                    with st.expander("Detail file yang gagal dihapus", expanded=False):
                        for item in delete_result["failed"]:
                            st.markdown(f"- `{item['path']}`")
                            st.caption(f"Alasan: {item['reason']}")
                st.rerun()
        else:
            st.caption("Belum ada file yang aman dihapus.")

        if unsafe_files:
            with st.expander("File yang belum aman dihapus", expanded=False):
                preview_unsafe = unsafe_files[:20]
                for path in preview_unsafe:
                    st.markdown(f"- `{path}`")
                if len(unsafe_files) > len(preview_unsafe):
                    st.caption(f"Masih ada {len(unsafe_files) - len(preview_unsafe)} file lain yang belum aman dihapus.")
                if skipped_unsafe_files:
                    st.caption("Sebagian file di atas termasuk file skipped yang perlu dikonversi dulu.")

        if latest_sync_report and (latest_sync_report.get("rows_skipped") or []):
            if st.button("Refresh Status File Skipped", key="refresh_skipped_status_button"):
                refreshed_report = dict(latest_sync_report)
                refreshed_report["rows_skipped"] = [
                    item for item in (latest_sync_report.get("rows_skipped") or []) if Path(item.get("path", "")).exists()
                ]
                save_sync_report(refreshed_report)
                st.session_state["last_sync_report"] = refreshed_report
                st.success("Status file skipped sudah diperbarui.")
                st.rerun()

        if st.button("Train Model dari feature store", key="train_model_button"):
            with st.spinner("Training model sedang berjalan..."):
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
                st.success("Training selesai. model.joblib sudah diperbarui.")
                if result.stdout.strip():
                    st.code(result.stdout.strip())
            else:
                st.error("Training gagal dijalankan.")
                if result.stdout.strip():
                    st.code(result.stdout.strip())
                if result.stderr.strip():
                    st.code(result.stderr.strip())


def render_how_it_works_panel():
    if DEMO_MODE:
        st.info("Mode demo online aktif. Halaman publik ini hanya menampilkan analisis dan penjelasan. Fitur dataset, training, dan pembersihan audio tetap khusus admin/lokal.")
    st.markdown('<div class="aa-article">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="aa-panel">
            <div class="aa-panel-title">How This Detector Works</div>
            <div class="aa-panel-subtitle">
                Penjelasan versi artikel untuk memahami cara kerja model, fitur yang dibaca, keterbatasan, dan cara
                menafsirkan hasil dashboard dengan jujur.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Audio AI Checker: How It Works")
    st.caption("Versi dokumentasi di dalam dashboard, ditulis agar lebih mudah dibaca seperti halaman penjelasan produk.")

    st.markdown(
        """
        <div class="aa-callout">
            <div class="aa-callout-title">Quick Reading Guide</div>
            <div>
                <strong>ID:</strong> Baca halaman ini sebagai panduan memahami keputusan dashboard, bukan sebagai janji bahwa model selalu benar.
                <br/>
                <strong>EN:</strong> Read this page as a guide to understand the dashboard's decisions, not as a promise that the model is always right.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="aa-summary-grid">
            <div class="aa-summary-card">
                <div class="aa-summary-card-title">Model Core</div>
                <p><strong>ID:</strong> Heuristik forensik + LogisticRegression terkalibrasi.</p>
                <p><strong>EN:</strong> Forensic heuristics + calibrated LogisticRegression.</p>
            </div>
            <div class="aa-summary-card">
                <div class="aa-summary-card-title">Decision Style</div>
                <p><strong>ID:</strong> Konservatif. Model boleh curiga, tetapi FAIL butuh bukti kuat.</p>
                <p><strong>EN:</strong> Conservative. The model may suspect AI, but FAIL still needs strong evidence.</p>
            </div>
            <div class="aa-summary-card">
                <div class="aa-summary-card-title">Training Data</div>
                <p><strong>ID:</strong> Belajar dari feature store agar pengetahuan lama tidak hilang.</p>
                <p><strong>EN:</strong> Learns from a feature store so older knowledge is preserved.</p>
            </div>
            <div class="aa-summary-card">
                <div class="aa-summary-card-title">Best Use</div>
                <p><strong>ID:</strong> Screening awal, audit manual, dan pengayaan dataset.</p>
                <p><strong>EN:</strong> Early screening, manual review, and dataset growth.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Tool ini mencoba menilai apakah audio lebih dekat ke pola Human, Hybrid, atau AI. "
        "Hasilnya sebaiknya dibaca sebagai screening assistant, bukan alat tuduhan otomatis."
    )

    st.markdown("### Bagaimana Model Bekerja")
    st.write(
        "Engine utama saat ini memakai kombinasi dua lapis: "
        "pertama, pembacaan heuristik forensik berbasis fitur audio; kedua, model ML untuk membaca pola fitur secara lebih global. "
        "Model ML yang dipakai sekarang adalah `LogisticRegression` yang dikalibrasi probabilitasnya, jadi bukan neural network besar."
    )
    st.caption(
        "EN: The engine uses two layers: forensic heuristics first, then an ML layer that reads the overall feature pattern. "
        "The current ML model is a calibrated `LogisticRegression`, not a large neural network."
    )
    st.write(
        "Karena itu, dashboard menampilkan dua sudut pandang sekaligus: "
        "`Forensic decision` untuk keputusan screening yang konservatif, dan `Prediksi ML` untuk melihat kecenderungan model."
    )
    st.caption(
        "EN: That is why the dashboard shows both `Forensic decision` for the conservative screening call and `Prediksi ML` for the model tendency."
    )

    st.markdown("### Training Process")
    st.write(
        "Model dilatih dari kumpulan audio berlabel `human`, `hybrid`, dan `ai`. "
        "Setiap file tidak langsung dipelajari sebagai waveform mentah, tetapi diubah dulu menjadi kumpulan angka fitur. "
        "Kumpulan fitur ini disimpan di `training_features.csv`, sehingga pengetahuan numerik lama bisa dipertahankan walaupun audio mentahnya nanti dibersihkan dari folder dataset."
    )
    st.caption(
        "EN: The model is trained from labeled `human`, `hybrid`, and `ai` audio. Each file is converted into numerical features first, "
        "and those features are stored in `training_features.csv` so older learning can survive even if raw audio is cleaned up later."
    )
    st.markdown(
        """
        - File audio diberi label yang benar melalui dashboard.
        - Fitur numerik disimpan ke feature store.
        - Training ulang model membaca feature store, bukan wajib dari audio lama.
        - Hasil training disimpan ke `model.joblib`.
        """
    )

    st.markdown("### Feature Analysis")
    st.write("Detector ini membaca beberapa kelompok fitur utama:")
    st.caption("EN: The detector reads several main feature groups.")
    st.markdown(
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

    st.markdown("### Cara Membaca Hasil")
    st.markdown(
        """
        - **PASS**: belum ada strong evidence yang cukup untuk menuduh AI.
        - **REVIEW**: ada hal yang perlu dicek manual, tapi bukti keras AI belum cukup.
        - **FAIL**: kombinasi bukti kuat sudah cukup serius dan audio patut dicurigai.
        """
    )
    st.caption(
        "EN: `PASS` means no strong evidence is present, `REVIEW` means manual checking is still needed, and `FAIL` means the strong evidence is serious enough to suspect AI."
    )
    st.write(
        "Kalau `Prediksi ML` tinggi ke AI tetapi `Forensic decision` masih `REVIEW`, itu berarti model curiga, "
        "namun bukti forensik kuatnya belum lengkap. Ini dibuat sengaja supaya dashboard lebih jujur dan tidak terlalu agresif."
    )
    st.caption(
        "EN: If the ML prediction leans strongly toward AI but the forensic decision is still `REVIEW`, the model is suspicious but the strong forensic evidence is still incomplete."
    )

    st.markdown("### Limitations and Considerations")
    st.markdown(
        """
        - Durasi audio yang sangat panjang bisa membuat pola repetisi terlihat lebih kuat dari yang sebenarnya.
        - Produksi modern dengan mastering padat, AutoTune berat, atau tekstur yang sangat rapi bisa terbaca mirip AI.
        - Lagu instrumental atau minim vokal membuat beberapa metrik vokal menjadi kurang informatif.
        - Model ini belajar dari dataset lokal yang terus berkembang, jadi kualitas label dataset sangat berpengaruh.
        - Hasil `archetype` generator tetap probabilistik dan sering wajar berakhir di `Unknown generator`.
        """
    )
    st.caption(
        "EN: Long audio, modern polished production, limited vocals, and dataset quality all affect reliability. Generator archetypes are still probabilistic and often remain `Unknown generator`."
    )

    st.markdown("### Next Steps")
    st.markdown(
        """
        - Menambah dataset yang lebih seimbang dan lebih bersih.
        - Memperkuat penjelasan awam agar user non-teknis lebih cepat paham.
        - Menambah visual training progress dan audit dataset yang lebih lengkap.
        - Mengembangkan fingerprint module agar semakin konsisten antara weak evidence, strong evidence, dan headline verdict.
        """
    )
    st.caption(
        "EN: Next improvements include a cleaner balanced dataset, clearer non-technical explanations, richer training visuals, and a more consistent fingerprint module."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_report(report: AnalysisReport, mode: str, y=None, sr=None, history_entry=None):
    st.divider()
    if history_entry is not None:
        st.info(
            f"Menampilkan hasil history untuk `{history_entry['source_name']}` dari {format_relative_time(history_entry['created_at'])}."
        )
        if history_entry.get("reference_link"):
            st.caption(f"Link referensi: {history_entry['reference_link']}")

    with st.container(border=True):
        st.markdown(f"### {report.overall_verdict}")
        st.caption(f"Forensic decision: {report.screening_outcome}")
        st.caption(f"Prediksi ML: {report.model_label}")
        st.caption(
            f"Headline probabilities: Human {report.headline_probabilities.human}% | "
            f"Hybrid {report.headline_probabilities.hybrid}% | "
            f"AI {report.headline_probabilities.ai}%"
        )
        st.caption(f"Engine: {report.analysis_engine}")
        st.caption(f"Keyakinan keputusan ({report.screening_outcome}): {report.verdict_confidence_label}")
        if report.verdict_confidence_reason:
            st.caption(report.verdict_confidence_reason)
        if report.decision_hint:
            st.caption(report.decision_hint)
        st.caption(f"Keyakinan untuk FAIL (AI tegas): {report.ai_verdict_confidence_label}")
        if report.ai_verdict_confidence_reason:
            st.caption(report.ai_verdict_confidence_reason)
        if report.fail_hint:
            st.caption(report.fail_hint)
        st.markdown(report.executive_summary)
        c1, c2, c3 = st.columns(3)
        c1.metric("Human", f"{report.headline_probabilities.human}%")
        c2.metric("Hybrid", f"{report.headline_probabilities.hybrid}%")
        c3.metric("Pure AI", f"{report.headline_probabilities.ai}%")
        st.caption(
            f"Angka di atas mengikuti basis verdict utama ({report.headline_probability_source}). "
            "Detail di bawah tetap menampilkan heuristik per domain dan keputusan model final."
        )

    # Create tabs for structured view
    tab_forensic, tab_ai_expert, tab_raw_data = st.tabs(["📊 Analisa Utama", "🤖 AI Auditor", "🛠️ Raw Data (JSON)"])

    with tab_forensic:
        with st.container(border=True):
            st.markdown("### Arti Hasil Ini")
            title_explanations = {
                "Likely Human Audio": "Judul ini berarti sistem tidak menemukan bukti kuat yang cukup untuk mencurigai AI.",
                "Possible AI Detected": "Judul ini berarti ada kombinasi bukti forensik yang cukup kuat sehingga audio patut dicurigai sebagai AI atau proses sintetis berat.",
                "AI Suspected (Review)": "Judul ini berarti model cukup curiga ke AI, tetapi keputusan akhirnya tetap review karena bukti forensik kuatnya belum lengkap.",
                "Likely Human (Review)": "Judul ini berarti audio cenderung human, tetapi masih ada konteks yang membuat sistem memilih review manual.",
                "Hybrid / Unclear (Review)": "Judul ini berarti hasilnya masih campuran atau belum cukup jelas, jadi sistem memilih review manual.",
            }
            forensic_explanations = {
                "PASS": "Ini keputusan forensik paling aman. Artinya tidak ada kombinasi bukti kuat yang cukup untuk menuduh AI.",
                "REVIEW": "Ini artinya file perlu dicek manual. Ada beberapa hal yang mencurigakan atau tidak stabil, tetapi bukti keras AI belum cukup.",
                "FAIL": "Ini artinya ada kombinasi bukti kuat yang cukup serius, jadi file patut dicurigai sebagai AI.",
            }
            model_explanations = {
                "Human": "Model ML paling condong membaca audio ini sebagai human.",
                "Hybrid": "Model ML paling condong membaca audio ini sebagai campuran atau berada di area abu-abu antara human dan AI-like pattern.",
                "AI": "Model ML paling condong membaca audio ini sebagai AI.",
            }
            fingerprint_explanations = {
                "Low": "Pola fingerprint generator tidak terlalu kuat.",
                "Medium": "Ada beberapa pola yang mirip audio generator, tetapi masih bisa tercampur dengan produksi modern.",
                "High": "Ada cukup banyak pola teknis yang mirip fingerprint audio generator atau audio yang diproses sangat rapi.",
            }

            st.markdown(f"- **Title:** {title_explanations.get(report.overall_verdict, 'Judul ini adalah ringkasan singkat dari pembacaan akhir sistem.')}")
            st.markdown(f"- **Forensic decision:** {forensic_explanations.get(report.screening_outcome, 'Ini adalah keputusan forensik utama sistem.')}")
            st.markdown(f"- **Prediksi ML:** {model_explanations.get(report.model_label, 'Ini adalah kelas yang paling dipilih oleh model ML.')}")
            st.markdown(
                f"- **Verdict probabilities:** Ini menunjukkan pembagian skor yang dipakai untuk judul dan keputusan utama: "
                f"Human {report.headline_probabilities.human}%, "
                f"Hybrid {report.headline_probabilities.hybrid}%, "
                f"AI {report.headline_probabilities.ai}%. "
                f"Di kasus ini angka headline memakai basis `{report.headline_probability_source}`. "
                "Detail di bawah tetap menampilkan probabilitas ML dan heuristik per domain."
            )
            st.markdown(
                f"- **Keyakinan keputusan:** Ini menunjukkan seberapa yakin sistem terhadap keputusan `{report.screening_outcome}`. "
                "Ini bukan berarti sistem yakin file ini pasti AI."
            )
            st.markdown("- **Keyakinan untuk FAIL (AI tegas):** Ini menunjukkan apakah bukti yang ada sudah cukup kuat untuk naik ke keputusan `FAIL`.")
            st.markdown(f"- **Fingerprint:** {fingerprint_explanations.get(report.fingerprint_level, 'Ini menunjukkan seberapa kuat pola generator-like yang terlihat.')}")
            st.markdown(
                f"- **Archetype:** Jika ada pola generator-like, sistem menilai file ini paling dekat ke `{report.fingerprint_archetype}`. "
                "Kalau tertulis `Unknown generator`, artinya pola itu belum cukup khas untuk diarahkan ke tipe generator tertentu."
            )

        with st.container(border=True):
            st.markdown("### Evidence Card")
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
                status = "Ya" if active else "Tidak"
                st.markdown(f"- **{indicator_labels.get(key, key)}:** {status}")

            if report.strong_red_flags:
                st.markdown("**Strong indicators:**")
                for item in report.strong_red_flags:
                    st.error(item)
            if report.weak_indicators:
                st.markdown("**Weak indicators:**")
                for item in report.weak_indicators:
                    st.warning(item)
            if report.production_mimic_indicators:
                st.markdown("**Production-mimic indicators:**")
                for item in report.production_mimic_indicators:
                    st.info(item)

        with st.container(border=True):
            st.markdown("### Fingerprint Module")
            st.markdown(f"**Generator-like fingerprint:** {report.fingerprint_level} ({report.fingerprint_score}/100)")
            st.markdown(f"**Most consistent with:** {report.fingerprint_archetype}")
            st.markdown(f"**Fingerprint confidence:** {report.fingerprint_confidence_label}")
            if report.fingerprint_confidence_reason:
                st.caption(report.fingerprint_confidence_reason)
            st.write(report.fingerprint_summary)
            if report.fingerprint_signals:
                for item in report.fingerprint_signals:
                    st.markdown(f"- {item}")
            if report.fingerprint_score_components:
                st.markdown("**Score breakdown:**")
                for key, value in report.fingerprint_score_components.items():
                    st.markdown(f"- `{key}`: `{value}`")
            if report.fingerprint_metrics:
                st.markdown("**Metrik pemicu fingerprint:**")
                for key, value in report.fingerprint_metrics.items():
                    st.markdown(f"- `{key}`: `{value}`")

        with st.container(border=True):
            st.markdown("### Bahasa Sederhana")
            st.write(report.simple_explanation)
            st.markdown(f"**Masalah utama:** {report.main_issue}")
            st.markdown(f"**Perbaiki di bagian:** {report.fix_area}")
            st.markdown("**Langkah praktis:**")
            for step in report.practical_steps:
                st.markdown(f"- {step}")

    with tab_ai_expert:
        with st.container(border=True):
            st.markdown("### 🤖 AI Expert Insight (Thinking Logic)")
            if report.expert_insight:
                # Handle potential error response from API
                if "error" in report.expert_insight:
                    st.error(report.expert_insight["error"])
                else:
                    # Executive Summary
                    st.markdown(f"**Executive Summary:**\n\n{report.expert_insight.get('executive_summary', '')}")
                    
                    # Findings & Production Mimics
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        st.markdown("**Top Findings:**")
                        for finding in report.expert_insight.get("top_findings", []):
                            st.markdown(f"✅ {finding}")
                    with col_f2:
                        st.markdown("**Could be Production (False Positives):**")
                        for mimic in report.expert_insight.get("could_be_production", []):
                            st.markdown(f"⚠️ {mimic}")
                    
                    # Manual Checks
                    st.markdown("**Manual Checks Needed:**")
                    for check in report.expert_insight.get("manual_checks", []):
                        st.markdown(f"- [ ] {check}")
                    
                    # Confidence Explainer
                    with st.expander("AI Reasoning & Gaps (Deep Logic)", expanded=False):
                        explainer = report.expert_insight.get("confidence_explainer", {})
                        st.markdown(f"**Reasoning:** {explainer.get('reasoning', '')}")
                        st.markdown(f"**Gap Analysis:** {explainer.get('gap', '')}")
            else:
                st.caption("Klik tombol di bawah untuk meminta modul AI (OpenAI) menganalisa data forensik ini lebih mendalam.")
                if st.button("🔍 Generate Expert Insight", key=f"gen_insight_{report.metadata.filename}"):
                    with st.spinner("Menghubungi AI Expert..."):
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
        st.markdown("### 🛠️ Raw Evidence Data (JSON)")
        st.markdown("**1. Copy System Prompt (Instruksi AI):**")
        st.code("Anda auditor forensik audio senior. Anda tidak mendengar audio, hanya membaca metrik. Jelaskan secara jujur. Jangan menyebut Suno/Udio sebagai kepastian kecuali ada metadata eksplisit atau fingerprint sangat kuat.", language="text")
        
        st.markdown("**2. Copy Data JSON (Metrik Teknis):**")
        st.caption("Copy JSON di bawah ini untuk ditempelkan ke ChatGPT atau alat analisa eksternal lainnya.")
        payload = get_minimal_payload(report)
        st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")
        st.info("💡 Data di atas adalah 'sidik jari' teknis lagu ini yang dikirim ke modul AI. Tidak mengandung audio mentah.")

    if history_entry is None and not DEMO_MODE:
        latest_upload = st.session_state.get("latest_analyzed_upload")
        if latest_upload:
            with st.container(border=True):
                st.markdown("### Simpan ke Dataset")
                st.caption("Pilih label yang benar jika file upload ini ingin dijadikan tambahan dataset training.")
                c1, c2, c3 = st.columns(3)
                if c1.button("Tambahkan ke Dataset Human", key="save_dataset_human"):
                    saved_path = add_uploaded_audio_to_dataset("human", latest_upload["name"], latest_upload["bytes"])
                    save_training_features("human", report, saved_path)
                    st.success(f"File disimpan ke dataset human: {saved_path.name}")
                if c2.button("Tambahkan ke Dataset Hybrid", key="save_dataset_hybrid"):
                    saved_path = add_uploaded_audio_to_dataset("hybrid", latest_upload["name"], latest_upload["bytes"])
                    save_training_features("hybrid", report, saved_path)
                    st.success(f"File disimpan ke dataset hybrid: {saved_path.name}")
                if c3.button("Tambahkan ke Dataset AI", key="save_dataset_ai"):
                    saved_path = add_uploaded_audio_to_dataset("ai", latest_upload["name"], latest_upload["bytes"])
                    save_training_features("ai", report, saved_path)
                    st.success(f"File disimpan ke dataset ai: {saved_path.name}")
                st.caption("Setelah beberapa file ditambahkan, kamu bisa train ulang model dari dataset yang sudah diperbarui.")

    st.markdown("### Detail")
    st.markdown(f"- Duration: {int(report.metadata.duration_sec)} seconds")
    st.markdown("- **Verdict probabilities (headline basis):**")
    st.markdown(f"  - Source used: {report.headline_probability_source}")
    st.markdown(f"  - {report.headline_probabilities.verdict_text}")
    st.markdown(f"  - Human: probability ({report.headline_probabilities.human}%)")
    st.markdown(f"  - Pure AI: probability ({report.headline_probabilities.ai}%)")
    st.markdown(f"  - Hybrid: probability ({report.headline_probabilities.hybrid}%)")

    st.markdown("- **ML probabilities (detail transparency):**")
    st.markdown(f"  - Engine used: {report.analysis_engine}")
    st.markdown(f"  - {report.final_probabilities.verdict_text}")
    st.markdown(f"  - Human: probability ({report.final_probabilities.human}%)")
    st.markdown(f"  - Pure AI: probability ({report.final_probabilities.ai}%)")
    st.markdown(f"  - Hybrid: probability ({report.final_probabilities.hybrid}%)")

    st.markdown("- **Heuristic combined probabilities:**")
    st.markdown(f"  - {report.heuristic_combined_analysis.verdict_text}")
    st.markdown(f"  - Human: probability ({report.heuristic_combined_analysis.human}%)")
    st.markdown(f"  - Pure AI: probability ({report.heuristic_combined_analysis.ai}%)")
    st.markdown(f"  - Hybrid: probability ({report.heuristic_combined_analysis.hybrid}%)")

    st.markdown("- **Heuristic spectral analysis:**")
    st.markdown(f"  - {report.spectral_analysis.verdict_text}")
    st.markdown(f"  - Human: probability ({report.spectral_analysis.human}%)")
    st.markdown(f"  - Pure AI: probability ({report.spectral_analysis.ai}%)")
    st.markdown(f"  - Hybrid: probability ({report.spectral_analysis.hybrid}%)")

    st.markdown("- **Heuristic temporal analysis:**")
    st.markdown(f"  - {report.temporal_analysis.verdict_text}")
    st.markdown(f"  - Human: probability ({report.temporal_analysis.human}%)")
    st.markdown(f"  - Pure AI: probability ({report.temporal_analysis.ai}%)")
    st.markdown(f"  - Hybrid: probability ({report.temporal_analysis.hybrid}%)")

    if len(report.limitations_warnings) > 0:
        st.markdown("---")
        st.markdown("**Peringatan & Rekomendasi:**")
        for warning in report.limitations_warnings:
            st.info(warning)
    if not MODEL_IS_RELIABLE:
        st.warning(MODEL_STATUS)
    elif report.confidence_label == "low":
        st.warning("Prediksi model untuk audio ini masih low confidence, jadi hasil akhir memakai heuristik.")

    if "Mode A" in mode:
        st.header("Mode A: Analisis Ultra-Akurat")
        tab1, tab2, tab3, tab4 = st.tabs(["1. Layer Metadata", "2. Metrik DSP Rinci", "3. Advanced Feature Parity", "4. Grafik"])

        with tab1:
            st.json(report.metadata.model_dump())
            if history_entry is not None and history_entry.get("reference_link"):
                st.caption(f"Link referensi: {history_entry['reference_link']}")
            if report.metadata.ai_tags_found:
                st.error(f"AI tags terdeteksi di metadata: {report.metadata.ai_tags_found}")
        with tab2:
            st.json(report.dsp.model_dump())
            st.markdown("### Red Flags")
            for item in report.red_flags:
                st.warning(item)
            if report.strong_red_flags:
                st.markdown("### Strong AI Indicators")
                for item in report.strong_red_flags:
                    st.error(item)
            if report.weak_indicators:
                st.markdown("### Weak Indicators")
                for item in report.weak_indicators:
                    st.warning(item)
            if report.production_mimic_indicators:
                st.markdown("### Production-Mimic Indicators")
                for item in report.production_mimic_indicators:
                    st.info(item)
            if report.ambiguous_red_flags:
                st.markdown("### Ambiguous Indicators")
                for item in report.ambiguous_red_flags:
                    st.warning(item)
            if report.normal_production_flags:
                st.markdown("### Normal Production Choices That Can Mimic AI")
                for item in report.normal_production_flags:
                    st.info(item)
        with tab3:
            st.markdown("### Matrix Parity Berdasarkan SubmitHub")

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
                st.info("Grafik tidak tersedia untuk item history karena source audio asli tidak disimpan.")

    elif "Mode C" in mode:
        st.header("Mode C: Humanization Editing Focus")
        st.markdown("Panduan langkah demi langkah untuk melakukan perbaikan di DAW, khususnya BandLab:")
        if len(report.humanization_guide) > 0:
            for idx, step in enumerate(report.humanization_guide, start=1):
                st.info(f"{idx}. {step}")
        else:
            st.success("Audio terlihat sudah natural. Tidak banyak humanisasi diperlukan.")

if DEMO_MODE:
    workspace_tab, how_it_works_tab = st.tabs(["Workspace", "How It Works"])
    with workspace_tab:
        st.caption("Area publik untuk upload audio, menjalankan analisis, dan membaca hasil screening secara cepat.")
        st.info("Online demo mode aktif. Training, dataset, dan history persisten disembunyikan agar penggunaan publik tetap aman.")
    with how_it_works_tab:
        render_how_it_works_panel()
else:
    workspace_tab, training_tab, history_tab, how_it_works_tab = st.tabs(
        ["Workspace", "Training & Dataset", "History", "How It Works"]
    )
    with workspace_tab:
        st.caption("Area kerja utama untuk menjalankan analisis dan melihat progres proses saat ini.")
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

