import os
from typing import Dict, Optional

import librosa
import mutagen
import numpy as np
import scipy.stats as stats

from models import (
    AnalysisProbabilities,
    AnalysisReport,
    AudioMetadata,
    DSPMetrics,
    SubmitHubMetrics,
)

AI_KEYWORDS = [
    "suno",
    "udio",
    "generated",
    "ai composer",
    "ai music",
    "voice clone",
    "ai ",
    "elevenlabs",
    "stable audio",
]


def _safe_float(value: Optional[float], default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _distribution_value(stats_obj, field: str) -> float:
    if stats_obj is None:
        return 0.0
    return _safe_float(getattr(stats_obj, field, 0.0))


def _append_unique(items: list[str], message: str) -> None:
    if message not in items:
        items.append(message)


def extract_metadata(filepath: str) -> AudioMetadata:
    info = mutagen.File(filepath)
    sample_rate = info.info.sample_rate if hasattr(info, "info") and hasattr(info.info, "sample_rate") else 44100
    channels = info.info.channels if hasattr(info, "info") and hasattr(info.info, "channels") else 2
    duration_sec = info.info.length if hasattr(info, "info") and hasattr(info.info, "length") else 0.0

    ai_tags = set()
    encoder = ""
    software = ""

    if info is not None and getattr(info, "tags", None) is not None:
        for key, value in info.tags.items():
            key_lower = key.lower()
            val_str = str(value).lower()
            if any(tag in key_lower or tag in val_str for tag in ["software", "encoder"]):
                if "software" in key_lower:
                    software = val_str
                if "encoder" in key_lower:
                    encoder = val_str
            for keyword in AI_KEYWORDS:
                if keyword in key_lower or keyword in val_str:
                    ai_tags.add(keyword)

    return AudioMetadata(
        filename=os.path.basename(filepath),
        sample_rate=sample_rate,
        channels=channels,
        duration_sec=duration_sec,
        encoder=encoder,
        software=software,
        ai_tags_found=sorted(ai_tags),
    )


def compute_dsp_metrics(y: np.ndarray, sr: int) -> tuple[DSPMetrics, SubmitHubMetrics]:
    peak_level = float(np.max(np.abs(y)))
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))

    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    dyn_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))

    if peak_level > 0.01:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        f0_valid = f0[f0 > 0]
        if len(f0_valid) > 0:
            f0_median = float(np.median(f0_valid))
            f0_std = float(np.std(f0_valid))
            pitch_diffs = np.abs(np.diff(f0_valid))
            pitch_jump_ratio = float(np.sum(pitch_diffs > 30) / len(pitch_diffs)) if len(pitch_diffs) > 0 else 0.0
        else:
            f0_median, f0_std, pitch_jump_ratio = None, None, None
    else:
        f0_median, f0_std, pitch_jump_ratio = None, None, None

    d_matrix = librosa.stft(y)
    spectrum = np.abs(d_matrix)
    cent_mean = float(np.mean(librosa.feature.spectral_centroid(S=spectrum, sr=sr)[0]))

    mean_spectrum = np.mean(spectrum, axis=1)
    total_energy = np.sum(mean_spectrum)
    normalized_spectrum = mean_spectrum / total_energy if total_energy > 0 else np.ones_like(mean_spectrum) / len(mean_spectrum)
    entropy_val = float(stats.entropy(normalized_spectrum))
    flatness_mean = float(np.mean(librosa.feature.spectral_flatness(S=spectrum)[0]))

    freqs = librosa.fft_frequencies(sr=sr)
    idx_8k = np.searchsorted(freqs, 8000)
    idx_12k = np.searchsorted(freqs, 12000)
    idx_20k = np.searchsorted(freqs, 20000)

    hf_8_12 = float(np.sum(mean_spectrum[idx_8k:idx_12k]) / total_energy) if total_energy > 0 else 0.0
    hf_12_20 = (
        float(np.sum(mean_spectrum[idx_12k:idx_20k] if idx_20k < len(mean_spectrum) else mean_spectrum[idx_12k:]) / total_energy)
        if total_energy > 0
        else 0.0
    )

    rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=spectrum, sr=sr, roll_percent=0.85)[0]))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean_list = np.mean(mfccs, axis=1).tolist()
    mfcc_std_list = np.std(mfccs, axis=1).tolist()
    spectral_flatness_median = float(np.median(librosa.feature.spectral_flatness(S=spectrum)[0]))

    y_harm, y_perc = librosa.effects.hpss(y)
    harm_ext = float(np.mean(librosa.feature.rms(y=y_harm)))
    perc_ext = float(np.mean(librosa.feature.rms(y=y_perc)))
    voc_music_ratio = harm_ext / (perc_ext + 1e-6)

    chroma_harm = librosa.feature.chroma_stft(y=y_harm, sr=sr)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    log_mel = librosa.power_to_db(mel_spect, ref=np.max)
    rms_frames = librosa.feature.rms(y=y_harm)

    def pool_features(feature_matrix: np.ndarray, blocks_sec: float = 0.5) -> np.ndarray:
        fps_local = sr / 512
        frames_per_block = max(1, int(blocks_sec * fps_local))
        total_frames = feature_matrix.shape[1]
        pooled_blocks = total_frames // frames_per_block
        if pooled_blocks == 0:
            return feature_matrix
        reshaped = feature_matrix[:, : pooled_blocks * frames_per_block].reshape(feature_matrix.shape[0], pooled_blocks, frames_per_block)
        mu = reshaped.mean(axis=2)
        sd = reshaped.std(axis=2)
        return np.vstack([mu, sd])

    chroma_pooled = pool_features(chroma_harm, 0.5)
    mel_pooled = pool_features(log_mel, 0.5)
    rms_pooled = pool_features(rms_frames, 0.5)[0, :]

    common_frames = min(chroma_pooled.shape[1], mel_pooled.shape[1], rms_pooled.shape[0])
    chroma_pooled = chroma_pooled[:, :common_frames]
    mel_pooled = mel_pooled[:, :common_frames]
    rms_pooled = rms_pooled[:common_frames]

    p60_thresh = np.percentile(rms_pooled, 60) if len(rms_pooled) > 0 else 0.0
    valid_blocks = rms_pooled > p60_thresh

    def cosine_pairwise_lag(feature_matrix: np.ndarray, lag_frames: int, valid_mask: np.ndarray, stride: int = 1) -> np.ndarray:
        total_frames = feature_matrix.shape[1]
        if total_frames <= lag_frames:
            return np.array([])
        idx = np.arange(0, total_frames - lag_frames, stride)
        valid_idx = [i for i in idx if valid_mask[i] and valid_mask[i + lag_frames]]
        if len(valid_idx) == 0:
            return np.array([])

        a_matrix = feature_matrix[:, valid_idx]
        b_matrix = feature_matrix[:, [i + lag_frames for i in valid_idx]]
        a_matrix = a_matrix / (np.linalg.norm(a_matrix, axis=0, keepdims=True) + 1e-9)
        b_matrix = b_matrix / (np.linalg.norm(b_matrix, axis=0, keepdims=True) + 1e-9)
        return np.sum(a_matrix * b_matrix, axis=0)

    def get_dist_stats_with_jitter(feature_matrix: np.ndarray, valid_mask: np.ndarray, lag_sec: float, jitter_sec: float, pool_sec: float = 0.5):
        fps_pooled = 1.0 / pool_sec
        lag_frames = int(lag_sec * fps_pooled)
        jitter_frames = max(1, int(jitter_sec * fps_pooled))
        if lag_frames - jitter_frames < 1:
            return None

        sim_base = cosine_pairwise_lag(feature_matrix, lag_frames, valid_mask, stride=1)
        if len(sim_base) == 0:
            return None

        sim_j_pos = cosine_pairwise_lag(feature_matrix, lag_frames + jitter_frames, valid_mask, stride=1)
        sim_j_neg = cosine_pairwise_lag(feature_matrix, lag_frames - jitter_frames, valid_mask, stride=1)
        med_base = float(np.median(sim_base))
        sim_jitter = np.concatenate([sim_j_pos, sim_j_neg]) if len(sim_j_neg) > 0 else sim_j_pos
        med_jitter = float(np.median(sim_jitter)) if len(sim_jitter) > 0 else med_base

        return {
            "median": med_base,
            "median_jitter": med_jitter,
            "p10": float(np.percentile(sim_base, 10)),
            "p90": float(np.percentile(sim_base, 90)),
            "over_0_98_ratio": float(np.sum(sim_base > 0.98) / len(sim_base)),
            "drop_vs_jitter": float(abs(med_base - med_jitter)),
        }

    def get_windowed_stats(feature_matrix: np.ndarray, valid_mask: np.ndarray) -> Dict[str, Optional[dict]]:
        stats_by_window = {"full": get_dist_stats_with_jitter(feature_matrix, valid_mask, lag_sec=5.0, jitter_sec=0.5)}
        for window_seconds, key in [(180, "w180s"), (120, "w120s"), (60, "w60s")]:
            stats_by_window[key] = get_dist_stats_with_jitter(feature_matrix, valid_mask, lag_sec=window_seconds, jitter_sec=3.0)
        return stats_by_window

    chroma_stats = get_windowed_stats(chroma_pooled, valid_blocks)
    mel_stats = get_windowed_stats(mel_pooled, valid_blocks)

    phases = np.angle(d_matrix)
    phase_diffs = np.diff(phases, axis=1)
    phase_coherence = 1.0 / (1.0 + float(np.std(phase_diffs)))

    spectrum_count = spectrum.shape[0]
    pooled_spectrum = pool_features(spectrum, 0.5)[:spectrum_count, :]
    pooled_common = min(pooled_spectrum.shape[1], len(valid_blocks))
    pooled_spectrum = pooled_spectrum[:, :pooled_common]
    valid_blocks_spectrum = valid_blocks[:pooled_common]

    idx_18k = np.searchsorted(freqs, 18000)
    hf_band_pooled = np.sum(pooled_spectrum[idx_12k:idx_18k, :], axis=0)
    total_band_pooled = np.sum(pooled_spectrum, axis=0) + 1e-10
    hf_shimmer_ratios = hf_band_pooled / total_band_pooled
    hf_shimmer_var = float(np.var(hf_shimmer_ratios))
    if len(rms_pooled) > 0:
        quiet_thresh = np.percentile(rms_pooled, 30)
        loud_thresh = np.percentile(rms_pooled, 70)
        quiet_mask = rms_pooled <= quiet_thresh
        loud_mask = rms_pooled >= loud_thresh
        quiet_hf = float(np.median(hf_shimmer_ratios[: len(rms_pooled)][quiet_mask])) if np.any(quiet_mask) else 0.0
        loud_hf = float(np.median(hf_shimmer_ratios[: len(rms_pooled)][loud_mask])) if np.any(loud_mask) else 0.0
        hf_quiet_loud_ratio = float(quiet_hf / (loud_hf + 1e-10)) if loud_hf > 0 else 0.0
    else:
        hf_quiet_loud_ratio = 0.0

    hf_band_spec = spectrum[idx_12k:idx_18k, :]
    hf_flux = np.sqrt(np.sum(np.diff(hf_band_spec, axis=1) ** 2, axis=0)) if hf_band_spec.shape[1] > 1 else np.array([0.0])
    hf_flux_variance = float(np.var(hf_flux))

    idx_200 = np.searchsorted(freqs, 200)
    idx_2k = np.searchsorted(freqs, 2000)
    mid_stats_60s = get_dist_stats_with_jitter(pooled_spectrum[idx_200:idx_2k, :], valid_blocks_spectrum, lag_sec=60.0, jitter_sec=3.0)
    mid_stats_120s = get_dist_stats_with_jitter(pooled_spectrum[idx_200:idx_2k, :], valid_blocks_spectrum, lag_sec=120.0, jitter_sec=3.0)
    air_stats_60s = get_dist_stats_with_jitter(pooled_spectrum[idx_12k:idx_18k, :], valid_blocks_spectrum, lag_sec=60.0, jitter_sec=3.0)

    dsp_metrics = DSPMetrics(
        peak_level=peak_level,
        rms_mean=rms_mean,
        dynamic_range_db=dyn_range,
        f0_median=f0_median,
        f0_std=f0_std,
        pitch_jump_ratio=pitch_jump_ratio,
        spectral_centroid_mean=cent_mean,
        spectral_entropy=entropy_val,
        spectral_flatness_mean=flatness_mean,
        hf_energy_ratio_8_12k=hf_8_12,
        hf_energy_ratio_12_20k=hf_12_20,
    )

    submithub_metrics = SubmitHubMetrics(
        spectral_rolloff_mean=rolloff,
        zero_crossing_rate_mean=zcr,
        mfcc_mean=mfcc_mean_list,
        mfcc_std=mfcc_std_list,
        phase_coherence=phase_coherence,
        harmonic_consistency_chroma=chroma_stats,
        texture_consistency_mel=mel_stats,
        spectral_flatness_median=spectral_flatness_median,
        hf_12k_20k_ratio=hf_12_20,
        hf_shimmer_variance=hf_shimmer_var,
        hf_quiet_loud_ratio=hf_quiet_loud_ratio,
        hf_flux_variance_highband=hf_flux_variance,
        mid_band_w60s=mid_stats_60s,
        mid_band_w120s=mid_stats_120s,
        air_band_w60s=air_stats_60s,
        voc_music_proxy_ratio=voc_music_ratio,
    )

    return dsp_metrics, submithub_metrics


def build_feature_vector(metadata: AudioMetadata, dsp: DSPMetrics, submithub: SubmitHubMetrics) -> Dict[str, float]:
    features = {
        "duration_sec": metadata.duration_sec,
        "sample_rate": float(metadata.sample_rate),
        "channels": float(metadata.channels),
        "metadata_ai_tag_count": float(len(metadata.ai_tags_found)),
        "peak_level": dsp.peak_level,
        "rms_mean": dsp.rms_mean,
        "dynamic_range_db": dsp.dynamic_range_db,
        "f0_median": _safe_float(dsp.f0_median),
        "f0_std": _safe_float(dsp.f0_std),
        "pitch_jump_ratio": _safe_float(dsp.pitch_jump_ratio),
        "spectral_centroid_mean": dsp.spectral_centroid_mean,
        "spectral_entropy": dsp.spectral_entropy,
        "spectral_flatness_mean": dsp.spectral_flatness_mean,
        "hf_energy_ratio_8_12k": dsp.hf_energy_ratio_8_12k,
        "hf_energy_ratio_12_20k": dsp.hf_energy_ratio_12_20k,
        "spectral_rolloff_mean": submithub.spectral_rolloff_mean,
        "zero_crossing_rate_mean": submithub.zero_crossing_rate_mean,
        "phase_coherence": submithub.phase_coherence,
        "spectral_flatness_median": submithub.spectral_flatness_median,
        "hf_shimmer_variance": submithub.hf_shimmer_variance,
        "voc_music_proxy_ratio": submithub.voc_music_proxy_ratio,
    }

    for idx, value in enumerate(submithub.mfcc_mean, start=1):
        features[f"mfcc_mean_{idx}"] = float(value)
    for idx, value in enumerate(submithub.mfcc_std, start=1):
        features[f"mfcc_std_{idx}"] = float(value)

    stats_map = {
        "chroma_full": submithub.harmonic_consistency_chroma.full,
        "chroma_w60s": submithub.harmonic_consistency_chroma.w60s,
        "chroma_w120s": submithub.harmonic_consistency_chroma.w120s,
        "chroma_w180s": submithub.harmonic_consistency_chroma.w180s,
        "mel_full": submithub.texture_consistency_mel.full,
        "mel_w60s": submithub.texture_consistency_mel.w60s,
        "mel_w120s": submithub.texture_consistency_mel.w120s,
        "mel_w180s": submithub.texture_consistency_mel.w180s,
        "mid_w60s": submithub.mid_band_w60s,
        "mid_w120s": submithub.mid_band_w120s,
        "air_w60s": submithub.air_band_w60s,
    }

    for prefix, stats_obj in stats_map.items():
        features[f"{prefix}_median"] = _distribution_value(stats_obj, "median")
        features[f"{prefix}_p10"] = _distribution_value(stats_obj, "p10")
        features[f"{prefix}_p90"] = _distribution_value(stats_obj, "p90")
        features[f"{prefix}_over_098_ratio"] = _distribution_value(stats_obj, "over_0_98_ratio")
        features[f"{prefix}_drop_vs_jitter"] = _distribution_value(stats_obj, "drop_vs_jitter")

    return features


def normalize_scores(scores: Dict[str, float]) -> Dict[str, int]:
    total = sum(scores.values())
    if total <= 0:
        return {"human": 33, "hybrid": 34, "ai": 33}

    raw = {label: (value / total) * 100.0 for label, value in scores.items()}
    rounded = {label: int(np.floor(value)) for label, value in raw.items()}
    remainder = 100 - sum(rounded.values())
    fractional_order = sorted(raw, key=lambda label: raw[label] - rounded[label], reverse=True)
    for label in fractional_order[:remainder]:
        rounded[label] += 1
    return rounded


def _build_probabilities(probs: Dict[str, float], label: str) -> AnalysisProbabilities:
    dominant = max(probs, key=probs.get)
    verdict = f"{label}: {dominant.title()} ({probs[dominant]}%)"
    return AnalysisProbabilities(
        human=probs["human"],
        hybrid=probs["hybrid"],
        ai=probs["ai"],
        verdict_text=verdict,
    )


def detect_generator_fingerprint(submithub: SubmitHubMetrics, dsp: DSPMetrics) -> dict:
    signals = []
    score = 0
    score_components = {
        "mel_tiling": 0,
        "mid_tiling": 0,
        "air_stability": 0,
        "hf_shimmer_stationarity": 0,
        "hf_quiet_loud": 0,
        "highband_transient_uniformity": 0,
    }

    mel_w60 = submithub.texture_consistency_mel.w60s
    mid_w60 = submithub.mid_band_w60s
    air_w60 = submithub.air_band_w60s

    if mel_w60 is not None and mel_w60.over_0_98_ratio > 0.80 and mel_w60.drop_vs_jitter < 0.01:
        score_components["mel_tiling"] = 30
        signals.append(
            f"Texture tiling di mel band kuat (over_0.98={mel_w60.over_0_98_ratio:.3f}, drop={mel_w60.drop_vs_jitter:.4f})."
        )
    elif mel_w60 is not None and mel_w60.over_0_98_ratio >= 0.45 and mel_w60.drop_vs_jitter <= 0.01:
        score_components["mel_tiling"] = 15
        signals.append(
            f"Texture tiling di mel band cukup jelas (over_0.98={mel_w60.over_0_98_ratio:.3f}, drop={mel_w60.drop_vs_jitter:.4f})."
        )

    if mid_w60 is not None and mid_w60.drop_vs_jitter < 0.03 and mid_w60.over_0_98_ratio > 0.10:
        score_components["mid_tiling"] = 20
        signals.append(
            f"Mid band menunjukkan tiling cukup konsisten (over_0.98={mid_w60.over_0_98_ratio:.3f}, drop={mid_w60.drop_vs_jitter:.4f})."
        )
    if air_w60 is not None and air_w60.median > 0.94 and air_w60.drop_vs_jitter < 0.01:
        score_components["air_stability"] = 15
        signals.append(
            f"Air band sangat konsisten antar bagian (median={air_w60.median:.3f}, drop={air_w60.drop_vs_jitter:.4f}, over_0.98={air_w60.over_0_98_ratio:.3f})."
        )
    if submithub.hf_shimmer_variance < 0.0002:
        score_components["hf_shimmer_stationarity"] = 15
        signals.append(f"HF shimmer sangat stasioner (variance={submithub.hf_shimmer_variance:.6f}).")
    elif submithub.hf_shimmer_variance <= 0.001:
        score_components["hf_shimmer_stationarity"] = 10
        signals.append(f"HF shimmer cenderung stasioner (variance={submithub.hf_shimmer_variance:.6f}).")

    if submithub.hf_quiet_loud_ratio >= 1.30:
        score_components["hf_quiet_loud"] = 20
        signals.append(
            f"Energi HF tetap kuat saat bagian quiet dibanding loud (quiet/loud={submithub.hf_quiet_loud_ratio:.3f})."
        )
    elif 0.85 <= submithub.hf_quiet_loud_ratio <= 1.15 and submithub.hf_12k_20k_ratio > 0.07:
        score_components["hf_quiet_loud"] = 10
        signals.append(
            f"HF tetap menyala mirip antara bagian quiet dan loud (quiet/loud={submithub.hf_quiet_loud_ratio:.3f})."
        )
    if submithub.hf_flux_variance_highband < 1.0:
        score_components["highband_transient_uniformity"] = 10
        signals.append(
            f"Transient band tinggi cenderung seragam (flux_variance={submithub.hf_flux_variance_highband:.3f})."
        )

    score = sum(score_components.values())
    score = min(score, 100)

    if score >= 70:
        level = "High"
    elif score >= 40:
        level = "Medium"
    else:
        level = "Low"

    if (
        level == "High"
        and mel_w60 is not None
        and mel_w60.over_0_98_ratio > 0.85
        and air_w60 is not None
        and 0.90 <= submithub.hf_quiet_loud_ratio <= 1.10
        and submithub.hf_shimmer_variance < 0.00015
        and submithub.hf_flux_variance_highband < 1.0
    ):
        archetype = "Suno-like"
    elif (
        level == "High"
        and air_w60 is not None
        and air_w60.median > 0.95
        and air_w60.drop_vs_jitter < 0.008
        and submithub.hf_flux_variance_highband < 1.0
    ):
        archetype = "Udio-like"
    else:
        archetype = "Unknown generator"

    if level == "Low":
        summary = "Fingerprint generator tidak cukup kuat."
    elif level == "Medium":
        summary = "Ada beberapa pola generator-like, tetapi masih bercampur dengan ciri produksi modern."
    else:
        summary = "Fingerprint generator cukup konsisten di beberapa modul spektral."

    if archetype == "Unknown generator":
        fingerprint_confidence = "low"
        fingerprint_reason = "Fingerprint teknis ada, tetapi belum cukup khas untuk diarahkan ke generator tertentu."
    elif level == "High":
        fingerprint_confidence = "medium"
        fingerprint_reason = "Pola fingerprint cukup konsisten, tetapi klasifikasi archetype masih berbasis rule konservatif."
    else:
        fingerprint_confidence = "low"
        fingerprint_reason = "Archetype masih lemah dan perlu dibaca hati-hati."

    return {
        "level": level,
        "archetype": archetype,
        "summary": summary,
        "signals": signals,
        "score": score,
        "score_components": score_components,
        "confidence": fingerprint_confidence,
        "confidence_reason": fingerprint_reason,
        "metrics": {
            "hf_shimmer_variance": round(submithub.hf_shimmer_variance, 6),
            "hf_quiet_loud_ratio": round(submithub.hf_quiet_loud_ratio, 4),
            "hf_flux_variance_highband": round(submithub.hf_flux_variance_highband, 4),
            "air_w60_median": round(air_w60.median, 4) if air_w60 is not None else 0.0,
            "air_w60_over_098_ratio": round(air_w60.over_0_98_ratio, 4) if air_w60 is not None else 0.0,
            "air_w60_drop_vs_jitter": round(air_w60.drop_vs_jitter, 4) if air_w60 is not None else 0.0,
            "mel_w60_over_098_ratio": round(mel_w60.over_0_98_ratio, 4) if mel_w60 is not None else 0.0,
            "mel_w60_drop_vs_jitter": round(mel_w60.drop_vs_jitter, 4) if mel_w60 is not None else 0.0,
            "mid_w60_over_098_ratio": round(mid_w60.over_0_98_ratio, 4) if mid_w60 is not None else 0.0,
            "mid_w60_drop_vs_jitter": round(mid_w60.drop_vs_jitter, 4) if mid_w60 is not None else 0.0,
        },
    }


def evaluate_audio(
    metadata: AudioMetadata,
    dsp: DSPMetrics,
    submithub: SubmitHubMetrics,
    feature_vector: Optional[Dict[str, float]] = None,
    ml_probabilities: Optional[Dict[str, float]] = None,
    analysis_engine: str = "heuristic-rules",
    confidence_label: str = "medium",
    confidence_reason: str = "",
) -> AnalysisReport:
    red_flags = []
    strong_red_flags = []
    ambiguous_red_flags = []
    normal_production_flags = []
    production_mimic_indicators = []
    weak_indicators = []
    green_flags = []

    spec_scores = {"human": 30.0, "hybrid": 15.0, "ai": 5.0}
    temp_scores = {"human": 30.0, "hybrid": 15.0, "ai": 5.0}

    if dsp.hf_energy_ratio_12_20k < 0.05:
        spec_scores["human"] += min(50.0, (0.05 - dsp.hf_energy_ratio_12_20k) * 1000.0)
        green_flags.append(f"Frekuensi tinggi cenderung natural (HF: {dsp.hf_energy_ratio_12_20k * 100:.1f}%).")
    elif dsp.hf_energy_ratio_12_20k > 0.12:
        spec_scores["ai"] += min(70.0, (dsp.hf_energy_ratio_12_20k - 0.12) * 800.0)
        ambiguous_red_flags.append("Energi frekuensi tinggi di atas 12kHz sangat menonjol.")
    else:
        spec_scores["hybrid"] += 20.0

    if dsp.spectral_entropy > 6.5:
        spec_scores["human"] += 2.0
    elif dsp.spectral_entropy < 2.5:
        spec_scores["ai"] += 3.0
        ambiguous_red_flags.append(f"Spectral entropy sangat rendah ({dsp.spectral_entropy:.1f}).")
    else:
        spec_scores["hybrid"] += 2.0

    chroma_stats = submithub.harmonic_consistency_chroma.full
    if chroma_stats is not None:
        if (
            chroma_stats.median > 0.97
            and chroma_stats.over_0_98_ratio > 0.5
            and chroma_stats.drop_vs_jitter < 0.03
            and submithub.hf_shimmer_variance < 0.0001
        ):
            spec_scores["ai"] += 50.0
            ambiguous_red_flags.append("Harmonic consistency sangat tinggi dan shimmer HF sangat stasioner.")
        elif chroma_stats.median > 0.97:
            spec_scores["hybrid"] += 8.0
            ambiguous_red_flags.append(f"Harmonic consistency sangat stabil ({chroma_stats.median:.3f}).")
        elif chroma_stats.median < 0.9:
            spec_scores["human"] += 6.0

    if dsp.dynamic_range_db > 25.0:
        temp_scores["human"] += min(40.0, (dsp.dynamic_range_db - 25.0) * 2.0)
        green_flags.append(f"Rentang dinamis lebar ({dsp.dynamic_range_db:.1f} dB).")
    elif dsp.dynamic_range_db < 15.0:
        temp_scores["ai"] += min(50.0, (15.0 - dsp.dynamic_range_db) * 4.0)
        normal_production_flags.append(f"Rentang dinamis sempit ({dsp.dynamic_range_db:.1f} dB) bisa juga datang dari mastering modern.")
        production_mimic_indicators.append(f"Dynamic range sempit ({dsp.dynamic_range_db:.1f} dB) masih sangat mungkin berasal dari mastering padat.")
    else:
        temp_scores["hybrid"] += 25.0

    if dsp.f0_std is not None:
        if dsp.f0_std > 20.0:
            temp_scores["human"] += min(40.0, (dsp.f0_std - 20.0) * 1.5)
        elif dsp.f0_std < 12.0:
            temp_scores["ai"] += min(50.0, (12.0 - dsp.f0_std) * 6.0)
            ambiguous_red_flags.append(f"Variasi pitch sangat stabil ({dsp.f0_std:.1f} Hz).")
            normal_production_flags.append("Pitch yang terlalu stabil juga bisa disebabkan AutoTune, Melodyne, atau editing vokal berat.")
        else:
            temp_scores["hybrid"] += 25.0

    if dsp.pitch_jump_ratio is not None:
        if dsp.pitch_jump_ratio > 0.05:
            temp_scores["human"] += min(30.0, (dsp.pitch_jump_ratio - 0.05) * 500.0)
        elif dsp.pitch_jump_ratio < 0.02:
            temp_scores["ai"] += min(30.0, (0.02 - dsp.pitch_jump_ratio) * 1000.0)
        else:
            temp_scores["hybrid"] += 15.0

    if submithub.phase_coherence > 0.9:
        temp_scores["ai"] += 6.0
        temp_scores["hybrid"] += 4.0
        ambiguous_red_flags.append("Phase coherence tinggi, tetapi indikator ini masih lemah jika berdiri sendiri.")
        production_mimic_indicators.append("Phase coherence tinggi bisa muncul dari kompresi, denoise, atau material yang tonal.")

    chroma_w60 = submithub.harmonic_consistency_chroma.w60s
    if chroma_w60 is not None and chroma_w60.over_0_98_ratio > 0.6:
        if chroma_w60.drop_vs_jitter < 0.03:
            temp_scores["ai"] += 50.0
            spec_scores["ai"] += 30.0
            ambiguous_red_flags.append(
                f"Pola 60 detik sangat repetitif dengan jitter drop kecil ({chroma_w60.drop_vs_jitter:.3f})."
            )
        elif chroma_w60.drop_vs_jitter > 0.08:
            temp_scores["hybrid"] += 30.0
            ambiguous_red_flags.append("Pola 60 detik repetitif, tapi masih berubah saat dijitter.")
            normal_production_flags.append("Repetisi bagian lagu juga bisa muncul normal pada chorus, loop DAW, atau genre repetitif.")
        else:
            temp_scores["hybrid"] += 15.0
            temp_scores["ai"] += 15.0
            ambiguous_red_flags.append("Ada pola repetitif jangka panjang, tetapi belum cukup kuat untuk menjadi fingerprint AI.")

    if metadata.ai_tags_found:
        spec_scores["ai"] += 500.0
        temp_scores["ai"] += 500.0
        ambiguous_red_flags.append("Metadata file mengandung tag yang mengarah ke AI.")

    red_flags = list(dict.fromkeys(strong_red_flags + ambiguous_red_flags))

    spec_res = normalize_scores(spec_scores)
    temp_res = normalize_scores(temp_scores)
    heuristic_final = normalize_scores(
        {
            "human": (spec_res["human"] + temp_res["human"]) / 2,
            "hybrid": (spec_res["hybrid"] + temp_res["hybrid"]) / 2,
            "ai": (spec_res["ai"] + temp_res["ai"]) / 2,
        }
    )

    final_res = normalize_scores(ml_probabilities) if ml_probabilities is not None else heuristic_final
    final_source = "model-calibrated" if ml_probabilities is not None else "heuristic-combined"
    dominant_label = max(final_res, key=final_res.get)
    model_label = {"human": "Human", "hybrid": "Hybrid", "ai": "AI"}[dominant_label]
    fingerprint = detect_generator_fingerprint(submithub, dsp)

    mel_w60 = submithub.texture_consistency_mel.w60s
    mel_w120 = submithub.texture_consistency_mel.w120s
    air_w60 = submithub.air_band_w60s
    mid_w60 = submithub.mid_band_w60s
    mid_w120 = submithub.mid_band_w120s

    s1a_texture_tiling = bool(
        mel_w60 is not None
        and mel_w60.over_0_98_ratio > 0.80
        and mel_w60.drop_vs_jitter < 0.01
    )
    s1b_chroma_tiling = bool(
        chroma_w60 is not None
        and chroma_w60.over_0_98_ratio > 0.20
        and chroma_w60.drop_vs_jitter < 0.03
    )
    s1b_mid_tiling = bool(
        mid_w60 is not None
        and mid_w120 is not None
        and mid_w60.over_0_98_ratio > 0.10
        and mid_w60.drop_vs_jitter < 0.03
        and mid_w120.over_0_98_ratio > 0.08
        and mid_w120.drop_vs_jitter < 0.03
    )
    s1b_structural_tiling = s1b_chroma_tiling or s1b_mid_tiling
    s1_tiling = s1a_texture_tiling and s1b_structural_tiling
    s2_air_stationary = bool(
        air_w60 is not None
        and submithub.hf_shimmer_variance < 0.0002
        and air_w60.over_0_98_ratio > 0.60
        and air_w60.drop_vs_jitter < 0.02
    )
    s2_air_stationary_weak = bool(
        air_w60 is not None
        and air_w60.median > 0.94
        and air_w60.drop_vs_jitter < 0.01
    )
    vocals_likely_present = submithub.voc_music_proxy_ratio > 1.5
    s3_vocal_lock = bool(
        vocals_likely_present
        and dsp.f0_std is not None
        and dsp.pitch_jump_ratio is not None
        and dsp.f0_std < 12.0
        and dsp.pitch_jump_ratio < 0.02
    )
    s4_metadata = bool(metadata.ai_tags_found)

    strong_indicator_status = {
        "S1a_texture_tiling": s1a_texture_tiling,
        "S1b_structural_tiling": s1b_structural_tiling,
        "S1_tiling_kuat": s1_tiling,
        "S2_weak_air_stability": s2_air_stationary_weak,
        "S2_air_band_stasioner": s2_air_stationary,
        "S3_vocal_lock_proxy": s3_vocal_lock,
        "S4_metadata_ai": s4_metadata,
    }
    strong_indicator_count = sum(
        [
            1 if s1_tiling else 0,
            1 if s2_air_stationary else 0,
            1 if s3_vocal_lock else 0,
            1 if s4_metadata else 0,
        ]
    )

    if s1_tiling:
        _append_unique(strong_red_flags, "S1: Long-range repetition sangat stabil dan tahan jitter, mirip pola tiling.")
    elif s1a_texture_tiling and not s1b_structural_tiling:
        weak_indicators.append("S1a aktif: texture tiling pada mel cukup kuat, tetapi belum ada bukti structural tiling dari chroma atau mid band.")
    elif s1b_structural_tiling and not s1a_texture_tiling:
        weak_indicators.append("S1b aktif: ada structural tiling ringan pada chroma atau mid band, tetapi texture tiling global belum cukup kuat.")
    elif (
        (chroma_w60 is not None and chroma_w60.over_0_98_ratio > 0.10)
        or (mel_w60 is not None and mel_w60.over_0_98_ratio > 0.35)
        or (mid_w60 is not None and mid_w60.over_0_98_ratio > 0.05)
    ):
        weak_indicators.append("Ada repetisi jangka panjang, tetapi belum cukup kuat atau belum cukup tahan jitter untuk disebut tiling.")

    if s2_air_stationary:
        _append_unique(strong_red_flags, "S2: HF shimmer sangat stasioner dan air band terlalu konsisten antar bagian.")
    elif s2_air_stationary_weak:
        weak_indicators.append("S2 weak aktif: air band cukup stabil, tetapi belum didukung oleh over_0.98 ratio yang kuat atau shimmer stationarity yang cukup.")
    elif air_w60 is not None and (air_w60.over_0_98_ratio > 0.30 or air_w60.drop_vs_jitter < 0.02):
        weak_indicators.append("Air band cukup stabil, tetapi sendirian belum cukup kuat untuk menuduh AI.")

    if s3_vocal_lock:
        _append_unique(strong_red_flags, "S3: Perilaku vokal sangat terkunci pada pitch proxy yang tersedia.")
    elif vocals_likely_present and (
        (dsp.f0_std is not None and dsp.f0_std < 12.0)
        or (dsp.pitch_jump_ratio is not None and dsp.pitch_jump_ratio < 0.02)
    ):
        weak_indicators.append("Ada indikasi pitch vokal sangat rapi, tetapi bukti vocal synthesis belum lengkap.")

    if s4_metadata:
        if "Metadata file mengandung tag yang mengarah ke AI." not in strong_red_flags:
            _append_unique(strong_red_flags, "S4: Metadata file mengandung tag yang mengarah ke AI.")

    if dsp.hf_energy_ratio_8_12k > 0.12:
        production_mimic_indicators.append("Energi 8-12 kHz cukup tinggi; ini bisa datang dari exciter, de-esser, hi-hat, atau sibilance.")
    if any(submithub.mfcc_mean[idx] > 20 for idx in (2, 3) if idx < len(submithub.mfcc_mean)):
        production_mimic_indicators.append("Sebagian MFCC mengarah ke karakter bright/air, tetapi ini lebih cocok dibaca sebagai warna produksi.")

    if final_res["ai"] >= 50 and strong_indicator_count < 2:
        weak_indicators.append("Model ML cukup curiga ke AI, tetapi bukti forensik kuat masih belum cukup.")
    if final_res["hybrid"] >= 40 and len(production_mimic_indicators) >= 2:
        weak_indicators.append("Model lebih melihat risiko hybrid karena ciri produksi modern cukup dominan.")

    strong_red_flags = list(dict.fromkeys(strong_red_flags))
    weak_indicators = list(dict.fromkeys(weak_indicators))
    ambiguous_red_flags = list(dict.fromkeys(ambiguous_red_flags))
    production_mimic_indicators = list(dict.fromkeys(production_mimic_indicators))
    normal_production_flags = list(dict.fromkeys(normal_production_flags))
    red_flags = list(dict.fromkeys(strong_red_flags + ambiguous_red_flags))

    effective_confidence_label = confidence_label
    effective_confidence_reason = confidence_reason
    if metadata.duration_sec > 180 and s1a_texture_tiling and not s1b_structural_tiling:
        if confidence_label == "high" and final_res["ai"] < 50 and strong_indicator_count == 0:
            effective_confidence_label = "medium"
            effective_confidence_reason = "Verdict confidence diturunkan karena lagu berdurasi panjang dan hanya S1a (mel-texture) yang aktif."
        elif confidence_label == "medium" and final_res["ai"] < 50 and strong_indicator_count == 0:
            effective_confidence_label = "low"
            effective_confidence_reason = "Verdict confidence diturunkan karena lagu berdurasi panjang dan hanya S1a (mel-texture) yang aktif."
        elif not confidence_reason:
            effective_confidence_reason = "Durasi panjang dengan S1a saja membuat pembacaan verdict perlu ditafsirkan lebih hati-hati."

    if strong_indicator_count >= 2 or (s4_metadata and (s1_tiling or s3_vocal_lock)):
        screening_outcome = "FAIL"
        overall = "Possible AI Detected"
    elif (
        strong_indicator_count == 1
        or final_res["ai"] >= 50
        or (final_res["hybrid"] >= 40 and len(production_mimic_indicators) >= 2)
    ):
        screening_outcome = "REVIEW"
        if model_label == "AI":
            overall = "AI Suspected (Review)"
        elif model_label == "Human":
            overall = "Likely Human (Review)"
        else:
            overall = "Hybrid / Unclear (Review)"
    elif final_res["ai"] < 40:
        screening_outcome = "PASS"
        overall = "Likely Human Audio"
    else:
        screening_outcome = "REVIEW"
        if model_label == "AI":
            overall = "AI Suspected (Review)"
        elif model_label == "Human":
            overall = "Likely Human (Review)"
        else:
            overall = "Hybrid / Unclear (Review)"

    if screening_outcome == "PASS":
        headline_probs = heuristic_final
        headline_source = "heuristic-combined"
    else:
        headline_probs = final_res
        headline_source = final_source

    if screening_outcome == "PASS" and headline_probs["ai"] > 30:
        screening_outcome = "REVIEW"
        overall = "Likely Human (Review)" if model_label == "Human" else "Hybrid / Unclear (Review)"
        headline_probs = final_res
        headline_source = final_source
    if screening_outcome == "REVIEW" and model_label == "AI" and headline_probs["ai"] < 40:
        overall = "Hybrid / Unclear (Review)"
    if screening_outcome == "FAIL" and headline_probs["ai"] < 40:
        screening_outcome = "REVIEW"
        overall = "Hybrid / Unclear (Review)"
        headline_probs = final_res
        headline_source = final_source

    if screening_outcome == "FAIL":
        ai_verdict_confidence_label = "high" if headline_probs["ai"] >= 60 and strong_indicator_count >= 2 else "medium"
        ai_verdict_confidence_reason = "Bukti forensik kuat sudah cukup untuk mendukung vonis AI." if ai_verdict_confidence_label == "high" else "Vonis AI didukung bukti kuat, tetapi masih perlu dibaca hati-hati."
    elif screening_outcome == "REVIEW":
        ai_verdict_confidence_label = "low"
        ai_verdict_confidence_reason = "Belum ada strong evidence yang cukup untuk FAIL, jadi vonis AI belum layak ditegaskan."
    else:
        ai_verdict_confidence_label = "low"
        ai_verdict_confidence_reason = "Keputusan saat ini tidak mengarah ke vonis AI."

    if screening_outcome == "REVIEW":
        ordered_probs = sorted(headline_probs.values(), reverse=True)
        margin = ordered_probs[0] - ordered_probs[1] if len(ordered_probs) > 1 else ordered_probs[0]
        decision_hint = (
            f"REVIEW: {effective_confidence_label} karena kelas headline teratas masih unggul {margin}% dan file tetap cukup mencurigakan untuk dicek manual."
        )
    elif screening_outcome == "PASS":
        decision_hint = f"PASS: {effective_confidence_label} karena strong evidence tetap rendah dan headline tidak cukup mendorong ke AI."
    else:
        decision_hint = f"FAIL: {effective_confidence_label} karena strong evidence terpenuhi dan headline cukup mendukung kecurigaan AI."

    if screening_outcome == "FAIL":
        fail_hint = f"FAIL: {ai_verdict_confidence_label} karena strong evidence {strong_indicator_count}/4 sudah cukup."
    else:
        fail_hint = f"FAIL: {ai_verdict_confidence_label} karena strong evidence belum terpenuhi ({strong_indicator_count}/4)."

    guide = []
    limitations = []

    if final_res["ai"] > 20 or final_res["hybrid"] > 40:
        guide.append("**Pitch & Formant:** Longgarkan retune speed agar vokal tidak terlalu terkunci.")
        guide.append("**Micro-Timing Shift:** Geser beberapa elemen beberapa milidetik untuk mematahkan timing yang terlalu kaku.")
        guide.append("**Volume Automation:** Tambahkan perubahan volume manual antar bagian agar dinamika terasa lebih hidup.")
        guide.append("**Saturation / Warmth:** Tambahkan saturasi ringan untuk menyamarkan kilap digital di frekuensi tinggi.")

    if chroma_stats is not None and chroma_stats.median > 0.8:
        guide.append("**Harmonic Humanization:** Variasikan velocity, timing, atau modulasi tipis agar harmoni tidak terlalu statis.")

    if any("Rentang dinamis sempit" in flag or "Variasi pitch sangat stabil" in flag for flag in (red_flags + normal_production_flags)):
        limitations.append("Produksi modern dengan kompresi berat, AutoTune, atau MIDI yang sangat rapi bisa terbaca mirip AI.")
    if metadata.duration_sec < 32 or metadata.duration_sec > 180:
        limitations.append("Durasi di luar rentang 32-180 detik cenderung membuat analisis kurang stabil.")
    if metadata.duration_sec > 180 and (s1a_texture_tiling or s1b_structural_tiling or s1_tiling):
        limitations.append("Pada durasi panjang, pola repetisi harus dibaca lebih hati-hati karena struktur lagu normal juga bisa tampak mirip antar bagian.")
    if "Possible AI Detected" != overall and final_res["hybrid"] >= 35:
        limitations.append("Kasus hybrid masih paling rentan tertukar dengan mixing modern atau editing vokal berat.")

    summary_parts = [f"Keputusan forensik saat ini adalah {screening_outcome} dengan verdict {overall.lower()}."]
    summary_parts.append(f"Prediksi ML paling condong ke kelas {model_label.lower()}.")
    summary_parts.append(f"Strong evidence: {strong_indicator_count}/4.")
    if strong_indicator_count > 0:
        summary_parts.append("Ada strong evidence yang aktif pada pemeriksaan utama.")
    elif fingerprint["level"] != "Low":
        summary_parts.append(
            f"Tidak ada strong evidence yang cukup untuk FAIL, tetapi fingerprint generator-like terdeteksi (tingkat: {fingerprint['level']})."
        )
    else:
        summary_parts.append("Tidak ada indikator AI kuat yang aktif pada empat pemeriksaan utama.")
    if weak_indicators:
        weak_context_labels = []
        if s1a_texture_tiling and not s1b_structural_tiling:
            weak_context_labels.append("S1a aktif")
        if s2_air_stationary_weak and not s2_air_stationary:
            weak_context_labels.append("S2 weak aktif")
        weak_context_text = ", ".join(weak_context_labels) if weak_context_labels else f"{len(weak_indicators)} sinyal lemah"
        summary_parts.append(f"Weak/context evidence: {weak_context_text}.")
    if production_mimic_indicators:
        summary_parts.append("Sebagian temuan lebih cocok dijelaskan oleh produksi modern daripada bukti AI keras.")
    executive_summary = " ".join(summary_parts)

    if screening_outcome == "PASS":
        simple_explanation = "Secara umum audio ini masih terbaca wajar. Sistem tidak menemukan kombinasi bukti kuat yang cukup untuk mencurigai AI."
    elif screening_outcome == "FAIL":
        simple_explanation = "Ada kombinasi bukti forensik yang cukup kuat, jadi audio ini patut dicurigai sebagai AI atau proses sintetis berat."
    else:
        simple_explanation = "Model atau aturan melihat beberapa hal yang mencurigakan, tetapi bukti kuatnya belum cukup untuk vonis keras, jadi hasilnya tetap review manual."

    if any("Rentang dinamis sempit" in flag for flag in normal_production_flags):
        main_issue = "Dinamika lagu terlalu rata atau terlalu padat."
        fix_area = "Mastering / mix bus"
        practical_steps = [
            "Kurangi limiter atau kompresor yang terlalu menekan di master.",
            "Biarkan verse dan chorus punya perbedaan energi yang lebih terasa.",
            "Cek apakah loudness terlalu dipaksa rata dari awal sampai akhir.",
        ]
    elif any("Variasi pitch sangat stabil" in flag for flag in ambiguous_red_flags):
        main_issue = "Vokal atau nada utama terdengar terlalu rapi dan terlalu stabil."
        fix_area = "Track vokal / pitch correction"
        practical_steps = [
            "Longgarkan AutoTune atau retune speed.",
            "Jangan ratakan semua nada secara terlalu presisi.",
            "Biarkan sedikit gerakan alami pada pitch dan timing vokal.",
        ]
    elif strong_red_flags:
        main_issue = "Ada indikator teknis yang cukup kuat dan sulit dijelaskan hanya dari mixing biasa."
        fix_area = "Sumber audio / arrangement / render awal"
        practical_steps = [
            "Bandingkan dengan stem asli atau premaster tanpa limiter.",
            "Cek apakah ada bagian yang terlalu copy-paste atau terlalu identik antar segmen.",
            "Pastikan sumber vokal dan instrumen benar-benar berasal dari take atau aransemen yang berbeda.",
        ]
    elif production_mimic_indicators:
        main_issue = "Temuan utama lebih mirip efek mixing atau mastering modern daripada bukti AI yang keras."
        fix_area = "Mastering, tone shaping, dan kerapian edit"
        practical_steps = [
            "Kurangi brightening atau exciter berlebih di area atas.",
            "Cek apakah limiter, de-esser, atau denoise terlalu agresif.",
            "Beri variasi kecil antar bagian agar tidak terasa terlalu seragam.",
        ]
    else:
        main_issue = "Belum ada masalah besar yang sangat spesifik, tetapi hasilnya masih perlu dibaca hati-hati."
        fix_area = "Review manual pada mix, mastering, dan vokal utama"
        practical_steps = [
            "Dengarkan ulang bagian verse, chorus, dan transisi untuk mencari bagian yang terlalu seragam.",
            "Bandingkan versi master dengan versi yang belum terlalu diproses jika ada.",
            "Prioritaskan pemeriksaan dinamika, vokal utama, dan variasi antar bagian lagu.",
        ]

    return AnalysisReport(
        metadata=metadata,
        dsp=dsp,
        submithub=submithub,
        feature_vector=feature_vector or build_feature_vector(metadata, dsp, submithub),
        spectral_analysis=_build_probabilities(spec_res, "Heuristic spectral score"),
        temporal_analysis=_build_probabilities(temp_res, "Heuristic temporal score"),
        heuristic_combined_analysis=_build_probabilities(heuristic_final, "Heuristic combined score"),
        final_probabilities=_build_probabilities(final_res, f"Final score ({final_source})"),
        headline_probabilities=_build_probabilities(headline_probs, f"Headline score ({headline_source})"),
        headline_probability_source=headline_source,
        overall_verdict=overall,
        model_label=model_label,
        screening_outcome=screening_outcome,
        executive_summary=executive_summary,
        simple_explanation=simple_explanation,
        main_issue=main_issue,
        fix_area=fix_area,
        practical_steps=practical_steps,
        analysis_engine=analysis_engine,
        confidence_label=effective_confidence_label,
        confidence_reason=effective_confidence_reason,
        verdict_confidence_label=effective_confidence_label,
        verdict_confidence_reason=effective_confidence_reason,
        ai_verdict_confidence_label=ai_verdict_confidence_label,
        ai_verdict_confidence_reason=ai_verdict_confidence_reason,
        decision_hint=decision_hint,
        fail_hint=fail_hint,
        fingerprint_confidence_label=fingerprint["confidence"],
        fingerprint_confidence_reason=fingerprint["confidence_reason"],
        fingerprint_level=fingerprint["level"],
        fingerprint_score=fingerprint["score"],
        fingerprint_archetype=fingerprint["archetype"],
        fingerprint_summary=fingerprint["summary"],
        fingerprint_signals=fingerprint["signals"],
        fingerprint_score_components=fingerprint["score_components"],
        fingerprint_metrics=fingerprint["metrics"],
        strong_indicator_status=strong_indicator_status,
        weak_indicators=weak_indicators,
        production_mimic_indicators=production_mimic_indicators,
        red_flags=red_flags,
        strong_red_flags=strong_red_flags,
        ambiguous_red_flags=ambiguous_red_flags,
        normal_production_flags=normal_production_flags,
        green_flags=green_flags,
        humanization_guide=guide,
        limitations_warnings=limitations,
    )
