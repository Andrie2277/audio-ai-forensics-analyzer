import librosa
import numpy as np
import scipy.stats as stats
import mutagen
import os
import random
from models import AudioMetadata, DSPMetrics, AnalysisReport, AnalysisProbabilities, SubmitHubMetrics

AI_KEYWORDS = ["suno", "udio", "generated", "ai composer", "ai music", "voice clone", "ai ", "elevenlabs", "stable audio"]

def extract_metadata(filepath: str) -> AudioMetadata:
    info = mutagen.File(filepath)
    sample_rate = info.info.sample_rate if hasattr(info, 'info') and hasattr(info.info, 'sample_rate') else 44100
    channels = info.info.channels if hasattr(info, 'info') and hasattr(info.info, 'channels') else 2
    duration_sec = info.info.length if hasattr(info, 'info') and hasattr(info.info, 'length') else 0.0
    
    ai_tags = set()
    encoder = ""
    software = ""
    
    if info is not None and getattr(info, 'tags', None) is not None:
        for key, value in info.tags.items():
            val_str = str(value).lower()
            if any(k in key.lower() or k in val_str for k in ['software', 'encoder']):
                if 'software' in key.lower():
                    software = val_str
                if 'encoder' in key.lower():
                    encoder = val_str
            for keyword in AI_KEYWORDS:
                if keyword in key.lower() or keyword in val_str:
                    ai_tags.add(keyword)
                    
    return AudioMetadata(
        filename=os.path.basename(filepath),
        sample_rate=sample_rate,
        channels=channels,
        duration_sec=duration_sec,
        encoder=encoder,
        software=software,
        ai_tags_found=list(ai_tags)
    )

def compute_dsp_metrics(y: np.ndarray, sr: int) -> tuple[DSPMetrics, SubmitHubMetrics]:
    peak_level = float(np.max(np.abs(y)))
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    dyn_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
    
    if peak_level > 0.01:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
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
        
    D = librosa.stft(y)
    S = np.abs(D)
    cent_mean = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)[0]))
    
    mean_spectrum = np.mean(S, axis=1)
    entropy_val = float(stats.entropy(mean_spectrum / np.sum(mean_spectrum)))
    flatness_mean = float(np.mean(librosa.feature.spectral_flatness(S=S)[0]))
    
    freqs = librosa.fft_frequencies(sr=sr)
    idx_8k = np.searchsorted(freqs, 8000)
    idx_12k = np.searchsorted(freqs, 12000)
    idx_20k = np.searchsorted(freqs, 20000)
    total_energy = np.sum(mean_spectrum)
    hf_8_12 = float(np.sum(mean_spectrum[idx_8k:idx_12k]) / total_energy) if total_energy > 0 else 0.0
    hf_12_20 = float(np.sum(mean_spectrum[idx_12k:idx_20k] if idx_20k < len(mean_spectrum) else mean_spectrum[idx_12k:]) / total_energy) if total_energy > 0 else 0.0

    # ----- NEW ROBUST SUBMITHUB FEATURES -----
    # 1. Basic Spectral
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean_list = np.mean(mfccs, axis=1).tolist()
    mfcc_std_list = np.std(mfccs, axis=1).tolist()
    
    spectral_flatness_median = float(np.median(librosa.feature.spectral_flatness(S=S)[0]))
    
    # 2. Harmonic Analysis (ROBUST)
    y_harm, y_perc = librosa.effects.hpss(y)
    
    # Vocals/Music proxy
    harm_ext = float(np.mean(librosa.feature.rms(y=y_harm)))
    perc_ext = float(np.mean(librosa.feature.rms(y=y_perc)))
    voc_music_ratio = harm_ext / (perc_ext + 1e-6)
    
    # Extract features for distribution analysis
    # Extract features for distribution analysis
    chroma_harm = librosa.feature.chroma_stft(y=y_harm, sr=sr)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    log_mel = librosa.power_to_db(mel_spect, ref=np.max)
    rms_frames = librosa.feature.rms(y=y_harm)
    
    # 3. Robust O(N) Pooling & Jittering Metrics
    def pool_features(F, blocks_sec=0.5):
        fps_local = sr / 512
        frames_per_block = max(1, int(blocks_sec * fps_local))
        T = F.shape[1]
        n = T // frames_per_block
        if n == 0: return F
        F_pool = F[:, :n*frames_per_block].reshape(F.shape[0], n, frames_per_block)
        mu = F_pool.mean(axis=2)
        sd = F_pool.std(axis=2)
        return np.vstack([mu, sd])

    chroma_pooled = pool_features(chroma_harm, 0.5)
    mel_pooled = pool_features(log_mel, 0.5)
    rms_pooled = pool_features(rms_frames, 0.5)[0, :] # Get mean RMS per block
    
    T_common = min(chroma_pooled.shape[1], mel_pooled.shape[1], rms_pooled.shape[0])
    chroma_pooled = chroma_pooled[:, :T_common]
    mel_pooled = mel_pooled[:, :T_common]
    rms_pooled = rms_pooled[:T_common]
    
    p60_thresh = np.percentile(rms_pooled, 60)
    valid_blocks = rms_pooled > p60_thresh
    
    def cosine_pairwise_lag(F, lag_frames, valid_mask, stride=1):
        T = F.shape[1]
        if T <= lag_frames: return np.array([])
        idx = np.arange(0, T - lag_frames, stride)
        valid_idx = [i for i in idx if valid_mask[i] and valid_mask[i + lag_frames]]
        if len(valid_idx) == 0: return np.array([])
        
        A = F[:, valid_idx]
        B = F[:, [i + lag_frames for i in valid_idx]]
        A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-9)
        B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-9)
        return np.sum(A * B, axis=0)

    def get_dist_stats_with_jitter(F_pooled, valid_mask, lag_sec=5.0, jitter_sec=0.5, pool_sec=0.5):
        fps_pooled = 1.0 / pool_sec
        lag_frames = int(lag_sec * fps_pooled)
        jitter_frames = max(1, int(jitter_sec * fps_pooled))
        
        if lag_frames - jitter_frames < 1:
            return None
            
        sim_base = cosine_pairwise_lag(F_pooled, lag_frames, valid_mask, stride=1)
        if len(sim_base) == 0: return None
        
        sim_j_pos = cosine_pairwise_lag(F_pooled, lag_frames + jitter_frames, valid_mask, stride=1)
        sim_j_neg = cosine_pairwise_lag(F_pooled, lag_frames - jitter_frames, valid_mask, stride=1)
        
        med_base = float(np.median(sim_base))
        sim_j_com = np.concatenate([sim_j_pos, sim_j_neg]) if len(sim_j_neg) > 0 else sim_j_pos
        med_jit = float(np.median(sim_j_com)) if len(sim_j_com) > 0 else med_base
        
        return {
            "median": med_base,
            "p10": float(np.percentile(sim_base, 10)),
            "p90": float(np.percentile(sim_base, 90)),
            "over_0_98_ratio": float(np.sum(sim_base > 0.98) / len(sim_base)),
            "drop_vs_jitter": float(abs(med_base - med_jit))
        }
        
    def get_windowed_stats(F_pooled, valid_mask):
        stats = {"full": get_dist_stats_with_jitter(F_pooled, valid_mask, lag_sec=5.0, jitter_sec=0.5)}
        for w_sec, key in [(180, "w180s"), (120, "w120s"), (60, "w60s")]:
            stats[key] = get_dist_stats_with_jitter(F_pooled, valid_mask, lag_sec=w_sec, jitter_sec=3.0)
        return stats

    chroma_stats = get_windowed_stats(chroma_pooled, valid_blocks)
    mel_stats = get_windowed_stats(mel_pooled, valid_blocks)

    phases = np.angle(D)
    phase_diffs = np.diff(phases, axis=1)
    phase_coherence = 1.0 / (1.0 + float(np.std(phase_diffs)))

    # 4. Stationary HF Shimmer Variance & Band-Split Texture
    features_count = S.shape[0]
    S_mean = pool_features(S, 0.5)[:features_count, :]
    
    T_common_S = min(S_mean.shape[1], len(valid_blocks))
    S_mean = S_mean[:, :T_common_S]
    valid_blocks_S = valid_blocks[:T_common_S]
    
    idx_18k = np.searchsorted(freqs, 18000)
    hf_band_pooled = np.sum(S_mean[idx_12k:idx_18k, :], axis=0)
    total_band_pooled = np.sum(S_mean, axis=0) + 1e-10
    hf_shimmer_ratios = hf_band_pooled / total_band_pooled
    hf_shimmer_var = float(np.var(hf_shimmer_ratios))
    
    idx_200 = np.searchsorted(freqs, 200)
    idx_2k = np.searchsorted(freqs, 2000)
    S_mid = S_mean[idx_200:idx_2k, :]
    S_air = S_mean[idx_12k:idx_18k, :]
    mid_stats_60s = get_dist_stats_with_jitter(S_mid, valid_blocks_S, lag_sec=60.0, jitter_sec=3.0)
    air_stats_60s = get_dist_stats_with_jitter(S_air, valid_blocks_S, lag_sec=60.0, jitter_sec=3.0)

    # Building Data Models
    dsp_metrics = DSPMetrics(
        peak_level=peak_level, rms_mean=rms_mean, dynamic_range_db=dyn_range,
        f0_median=f0_median, f0_std=f0_std, pitch_jump_ratio=pitch_jump_ratio,
        spectral_centroid_mean=cent_mean, spectral_entropy=entropy_val, spectral_flatness_mean=flatness_mean,
        hf_energy_ratio_8_12k=hf_8_12, hf_energy_ratio_12_20k=hf_12_20
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
        mid_band_w60s=mid_stats_60s,
        air_band_w60s=air_stats_60s,
        
        voc_music_proxy_ratio=voc_music_ratio
    )


    return dsp_metrics, submithub_metrics

def normalize_scores(scores: dict) -> dict:
    total = sum(scores.values())
    if total == 0:
        return {"human": 33, "hybrid": 34, "ai": 33}
    return {k: round((v / total) * 100) for k, v in scores.items()}

def evaluate_audio(metadata: AudioMetadata, dsp: DSPMetrics, submithub: SubmitHubMetrics) -> AnalysisReport:
    red_flags = []
    green_flags = []
    
    # --- SPECTRAL CONTINUOUS SCORING ---
    spec_scores = {"human": 30.0, "hybrid": 15.0, "ai": 5.0}
    
    # Map HF Energy (12k-20k). Human < 0.05, AI > 0.12
    if dsp.hf_energy_ratio_12_20k < 0.05:
        add_human = min(50.0, (0.05 - dsp.hf_energy_ratio_12_20k) * 1000.0)
        spec_scores["human"] += add_human
        green_flags.append(f"Frekuensi tinggi sangat natural (HF: {dsp.hf_energy_ratio_12_20k*100:.1f}%)")
    elif dsp.hf_energy_ratio_12_20k > 0.12:
        add_ai = min(70.0, (dsp.hf_energy_ratio_12_20k - 0.12) * 800.0)
        spec_scores["ai"] += add_ai
        red_flags.append(f"Pola shimmer frekuensi tinggi janggal (>12kHz): indikasi kuat buatan AI.")
    else:
        spec_scores["hybrid"] += 20.0
        
    # Map Spectral Entropy
    if dsp.spectral_entropy is not None:
        if dsp.spectral_entropy > 6.0:
            spec_scores["human"] += 5.0
        elif dsp.spectral_entropy < 3.0:
            spec_scores["ai"] += 10.0
            red_flags.append(f"Spectral Entropy tergolong rendah ({dsp.spectral_entropy:.1f}). Mungkin spektrum AI sintesis atau pad berlapis tebal.")
        else:
            spec_scores["hybrid"] += 5.0

    # SubmitHub Advanced Spectral
    chroma_stats = submithub.harmonic_consistency_chroma.full
    if chroma_stats is not None:
        if chroma_stats.median > 0.97 and submithub.hf_shimmer_variance < 0.0001:
            spec_scores["ai"] += 50.0
            red_flags.append(f"Harmonic Cosine Sim ekstrim ({chroma_stats.median:.3f}) dan Shimmer HF stasioner. Indikasi kuat looping buatan AI.")
        elif chroma_stats.median > 0.97:
            spec_scores["hybrid"] += 15.0
            red_flags.append(f"Harmonic Consistency sangat stabil ({chroma_stats.median:.3f}). Bisa jadi susunan chord statis atau DAW loop.")
        else:
            spec_scores["human"] += 15.0

    # --- TEMPORAL CONTINUOUS SCORING ---
    temp_scores = {"human": 30.0, "hybrid": 15.0, "ai": 5.0}
    
    # Map Dynamic Range. Natural > 25dB, AI Brickwall < 15dB
    if dsp.dynamic_range_db > 25.0:
        add_human = min(40.0, (dsp.dynamic_range_db - 25.0) * 2.0)
        temp_scores["human"] += add_human
        green_flags.append(f"Rentang dinamis natural/lebar ({dsp.dynamic_range_db:.1f} dB).")
    elif dsp.dynamic_range_db < 15.0:
        add_ai = min(50.0, (15.0 - dsp.dynamic_range_db) * 4.0)
        temp_scores["ai"] += add_ai
        red_flags.append(f"Rentang dinamis terlalu rata/sempit ({dsp.dynamic_range_db:.1f} dB), indikasi *brickwalling* AI.")
    else:
        temp_scores["hybrid"] += 25.0
        
    # Map Pitch Stability (f0_std). Natural > 20Hz, AI < 12Hz
    if dsp.f0_std is not None:
        if dsp.f0_std > 20.0:
            temp_scores["human"] += min(40.0, (dsp.f0_std - 20.0) * 1.5)
        elif dsp.f0_std < 12.0:
            temp_scores["ai"] += min(50.0, (12.0 - dsp.f0_std) * 6.0)
            red_flags.append(f"Variasi pitch/nada terlalu stabil ({dsp.f0_std:.1f} Hz). Lacks human micro-modulation.")
        else:
            temp_scores["hybrid"] += 25.0
            
    # Pitch Jump Ratio
    if dsp.pitch_jump_ratio is not None:
        if dsp.pitch_jump_ratio > 0.05:
            temp_scores["human"] += min(30.0, (dsp.pitch_jump_ratio - 0.05) * 500.0)
        elif dsp.pitch_jump_ratio < 0.02:
            temp_scores["ai"] += min(30.0, (0.02 - dsp.pitch_jump_ratio) * 1000.0)
        else:
            temp_scores["hybrid"] += 15.0

    # SubmitHub Advanced Temporal
    if submithub.phase_coherence > 0.85:
        temp_scores["ai"] += 20.0
        temp_scores["hybrid"] += 10.0
        
    # Long Range & Texture Variance Check
    chroma_w60 = submithub.harmonic_consistency_chroma.w60s
    if chroma_w60 is not None:
        if chroma_w60.over_0_98_ratio > 0.6:
            if chroma_w60.drop_vs_jitter < 0.03:
                temp_scores["ai"] += 50.0
                spec_scores["ai"] += 30.0
                red_flags.append(f"Long-Range W60s terdeteksi *AI Tiling* (Drop < 0.03: {chroma_w60.drop_vs_jitter:.3f}). Jitter gagal merusak loop matematikal.")
            elif chroma_w60.drop_vs_jitter > 0.08:
                temp_scores["hybrid"] += 30.0
                red_flags.append(f"Pola W60s repetitif tapi memiliki Anti-Bar Jitter Drop besar (>0.08). Indikasi *Copy-Paste Loop* DAW.")
            else:
                temp_scores["hybrid"] += 15.0
                temp_scores["ai"] += 15.0
                if submithub.hf_shimmer_variance < 0.0001:
                    spec_scores["ai"] += 30.0
                    red_flags.append(f"Pola W60s ambigu (Drop={chroma_w60.drop_vs_jitter:.3f}) TAPI Shimmer HF sangat stasioner (var: {submithub.hf_shimmer_variance:.6f}). Cenderung AI.")
                else:
                    red_flags.append(f"Pola W60s ambigu (Drop={chroma_w60.drop_vs_jitter:.3f}). Disarankan cek stem manual.")
            
    mel_stats = submithub.texture_consistency_mel.full
    if chroma_stats is not None and mel_stats is not None:
        if chroma_stats.p90 > 0.98 and mel_stats.p90 > 0.98:
            temp_scores["ai"] += 50.0
            spec_scores["ai"] += 30.0
            red_flags.append("Tekstur melogram dan harmoni beresonansi identik tinggi (P90 > 0.98). Kombinasi fatal AI generator.")

    # Metadata Overrides
    if metadata.ai_tags_found:
        spec_scores["ai"] += 500.0
        temp_scores["ai"] += 500.0
        red_flags.append("Metadata secara eksplisit menyatakan ini adalah produk AI.")

    spec_res = normalize_scores(spec_scores)
    temp_res = normalize_scores(temp_scores)

    def get_verdict(probs):
        m = max(probs, key=probs.get)
        if probs["ai"] > 50: return f"Pure AI: highly likely ({probs['ai']}%)"
        if probs["human"] > 50: return f"Human: most likely ({probs['human']}%)"
        if m == "hybrid": return f"Hybrid (AI + Human): most likely ({probs['hybrid']}%)"
        return f"{m.title()}: could be ({probs[m]}%)"

    spec_prob = AnalysisProbabilities(
        human=spec_res["human"],
        hybrid=spec_res["hybrid"],
        ai=spec_res["ai"],
        verdict_text=get_verdict(spec_res)
    )
    
    temp_prob = AnalysisProbabilities(
        human=temp_res["human"],
        hybrid=temp_res["hybrid"],
        ai=temp_res["ai"],
        verdict_text=get_verdict(temp_res)
    )
    
    ai_avg = (spec_res["ai"] + temp_res["ai"]) / 2
    human_avg = (spec_res["human"] + temp_res["human"]) / 2
    hybrid_avg = (spec_res["hybrid"] + temp_res["hybrid"]) / 2
    
    if metadata.ai_tags_found or (ai_avg > 60):
        overall = "Possible AI Detected 🤖"
    elif hybrid_avg > 50 or (ai_avg > 30 and ai_avg <= 60):
        overall = "Hybrid Audio / Unclear 🟡"
    else:
        overall = "Likely Human Audio ✅"

    guide = []
    limitations = []
    
    if ai_avg > 20 or hybrid_avg > 40:
        guide.append("**Pitch & Formant:** Use BandLab AutoPitch to *loosen* the retune speed. Too fast = robotic. If vocals sound 'locked', detune specific notes manually slightly.")
        guide.append("**Micro-Timing Shift:** Turn off 'Snap to Grid' in BandLab. Nudge drum or vocal clips a few milliseconds left/right to break rigid AI timing.")
        guide.append("**Volume Automation:** Draw volume curves manually for choruses/verses to simulate breath and energy changes. The original track's dynamics were too flat.")
        guide.append("**Saturation / Warmth:** Apply 'Tube Screamer' or 'Tape Saturator' presets lightly to the mix bus to hide artificial digital high-frequency shimmer.")
        
    if submithub.harmonic_consistency_chroma.full is not None:
        if submithub.harmonic_consistency_chroma.full.median > 0.8:
            guide.append("**Harmonic Humanization:** Kurangi 'kesempurnaan' chord Anda. Jika menggunakan MIDI, gunakan fitur *Humanize* (acak *Velocity* dan geser not sepersekian milidetik secara acak). Jika instrumen asli, beri sedikit efek *Modulation* (seperti *Chorus* ringan atau *Wow/Flutter* tape) agar harmoni tidak terlalu statis/konstan.")


    
    if any("narrow dynamic range" in r or "Pitch is extremely stable" in r for r in red_flags):
        limitations.append("Awas False Positive: Pilihan produksi modern (AutoTune tebal, kompresi mastering berat, sampel drum MIDI) bisa terbaca seperti pola AI. Sangat disarankan untuk meminta **Dry Vocal Stem** atau **Premaster Tanpa Limiter**.")
        
    if "Hybrid" in overall:
        limitations.append("Audio ini tidak menunjukkan *generator fingerprint* yang mutlak, tapi tetap mencurigakan. Sulit menentukan jika ini hanya efek Melodyne/De-Esser yang kencang atau memang buatan AI.")

    return AnalysisReport(
        metadata=metadata,
        dsp=dsp,
        submithub=submithub,
        spectral_analysis=spec_prob,
        temporal_analysis=temp_prob,
        overall_verdict=overall,
        red_flags=red_flags,
        green_flags=green_flags,
        humanization_guide=guide,
        limitations_warnings=limitations
    )


# Legacy compatibility override:
# keep old file importable, but route active calls to analyzer_ml.
from analyzer_ml import (  # noqa: E402
    build_feature_vector as build_feature_vector,
    compute_dsp_metrics as compute_dsp_metrics,
    evaluate_audio as evaluate_audio,
    extract_metadata as extract_metadata,
)
