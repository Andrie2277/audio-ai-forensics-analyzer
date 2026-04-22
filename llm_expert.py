import os
import json
from typing import Optional, Dict
from openai import OpenAI
from dotenv import load_dotenv
from models import AnalysisReport

# Load environment variables (API Key)
load_dotenv()


def get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
Anda adalah Auditor Forensik Audio Senior. Anda tidak mendengar audio, hanya membaca metrik teknis. 
Jelaskan secara jujur dan berikan analisa mendalam. 
Jangan menyebut Suno/Udio sebagai kepastian kecuali ada metadata eksplisit atau fingerprint sangat kuat.

Output Anda HARUS dalam format JSON dengan struktur berikut:
{
  "executive_summary": "Ringkasan singkat analisa Anda (maks 220 kata).",
  "top_findings": ["Temuan kunci 1", "Temuan kunci 2", "Temuan kunci 3"],
  "could_be_production": ["Potensi false positive 1", "Potensi false positive 2", "Potensi false positive 3"],
  "manual_checks": ["Langkah cek manual 1", "Langkah cek manual 2"],
  "confidence_explainer": {
    "reasoning": "Penjelasan kenapa mesin memberikan verdict tersebut.",
    "gap": "Kenapa hasilnya tidak lebih buruk atau tidak lebih baik (misal: kenapa REVIEW bukan FAIL)."
  }
}

Gunakan Bahasa Indonesia yang profesional, teknis, namun tetap lugas. Gunakan bullet points di dalam string jika diperlukan.
"""

def get_minimal_payload(report: AnalysisReport) -> Dict:
    """
    Menyusun payload 'Minimal' sesuai spesifikasi Payload Minimal.txt
    """
    return {
        "file": {
            "name": report.metadata.filename,
            "duration_sec": report.metadata.duration_sec,
            "sample_rate": report.metadata.sample_rate,
            "channels": report.metadata.channels
        },
        "headline": {
            "title": report.overall_verdict,
            "screening_outcome": report.screening_outcome,
            "headline_probs": report.headline_probabilities.model_dump(),
            "prediksi_ml": report.model_label
        },
        "confidence": {
            "review_confidence": report.verdict_confidence_label,
            "fail_confidence": report.ai_verdict_confidence_label,
            "review_reason": report.verdict_confidence_reason,
            "fail_reason": report.ai_verdict_confidence_reason
        },
        "evidence": {
            "strong": report.strong_indicator_status,
            "weak": report.weak_indicators,
            "production_mimic": report.production_mimic_indicators
        },
        "fingerprint_metrics": report.fingerprint_metrics,
        "parity": {
            "mel_w60": {
                "over_098": report.submithub.texture_consistency_mel.w60s.over_0_98_ratio if report.submithub.texture_consistency_mel.w60s else 0,
                "drop": report.submithub.texture_consistency_mel.w60s.drop_vs_jitter if report.submithub.texture_consistency_mel.w60s else 0
            },
            "chroma_w60": {
                "drop": report.submithub.harmonic_consistency_chroma.w60s.drop_vs_jitter if report.submithub.harmonic_consistency_chroma.w60s else 0
            },
            "mid_w60": {
                "drop": report.submithub.mid_band_w60s.drop_vs_jitter if report.submithub.mid_band_w60s else 0,
                "over_098": report.submithub.mid_band_w60s.over_0_98_ratio if report.submithub.mid_band_w60s else 0
            },
            "air_w60": {
                "median": report.submithub.air_band_w60s.median if report.submithub.air_band_w60s else 0,
                "drop": report.submithub.air_band_w60s.drop_vs_jitter if report.submithub.air_band_w60s else 0,
                "over_098": report.submithub.air_band_w60s.over_0_98_ratio if report.submithub.air_band_w60s else 0
            }
        },
        "dsp": {
            "dynamic_range_db": report.dsp.dynamic_range_db,
            "spectral_entropy": report.dsp.spectral_entropy,
            "spectral_flatness_mean": report.dsp.spectral_flatness_mean,
            "hf_energy_ratio_8_12k": report.dsp.hf_energy_ratio_8_12k,
            "hf_energy_ratio_12_20k": report.dsp.hf_energy_ratio_12_20k
        }
    }

def generate_expert_insight(report: AnalysisReport) -> Optional[Dict]:
    """
    Mengirimkan payload 'Minimal' ke OpenAI dan mendapatkan respons JSON terstruktur.
    """
    client = get_openai_client()
    if client is None:
        return {"error": "API Key OpenAI tidak ditemukan di file .env"}

    payload = get_minimal_payload(report)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Berikut ringkasan metrik analisis audio. Tugas Anda: Jelaskan kenapa hasilnya {report.screening_outcome}, sebutkan 3 temuan penting, 3 potensi false positive, dan langkah cek manual.\n\nDATA: {json.dumps(payload)}"}
            ],
            response_format={"type": "json_object"},
            max_tokens=1000,
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": f"Gagal mendapatkan insight dari AI: {str(e)}"}
