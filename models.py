from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, List, Optional

class AudioMetadata(BaseModel):
    model_config = ConfigDict(strict=False)
    
    filename: str
    sample_rate: int
    channels: int
    duration_sec: float
    bit_depth: Optional[int] = None
    encoder: Optional[str] = None
    software: Optional[str] = None
    ai_tags_found: List[str] = Field(default_factory=list)

class DSPMetrics(BaseModel):
    model_config = ConfigDict(strict=False)
    
    peak_level: float
    rms_mean: float
    dynamic_range_db: float
    f0_median: Optional[float] = None
    f0_std: Optional[float] = None
    pitch_jump_ratio: Optional[float] = None
    spectral_centroid_mean: float
    spectral_entropy: float
    spectral_flatness_mean: float
    hf_energy_ratio_8_12k: float
    hf_energy_ratio_12_20k: float

class DistributionStats(BaseModel):
    median: float
    median_jitter: float
    p10: float
    p90: float
    over_0_98_ratio: float
    drop_vs_jitter: float

class WindowedHarmonicStats(BaseModel):
    full: Optional[DistributionStats] = None
    w180s: Optional[DistributionStats] = None
    w120s: Optional[DistributionStats] = None
    w60s: Optional[DistributionStats] = None

class SubmitHubMetrics(BaseModel):
    model_config = ConfigDict(strict=False)
    
    spectral_rolloff_mean: float
    zero_crossing_rate_mean: float
    mfcc_mean: List[float]
    mfcc_std: List[float]
    phase_coherence: float
    
    # New robust metrics
    harmonic_consistency_chroma: WindowedHarmonicStats
    texture_consistency_mel: WindowedHarmonicStats
    
    spectral_flatness_median: float
    hf_12k_20k_ratio: float
    hf_shimmer_variance: float
    hf_quiet_loud_ratio: float = 0.0
    hf_flux_variance_highband: float = 0.0
    mid_band_w60s: Optional[DistributionStats] = None
    mid_band_w120s: Optional[DistributionStats] = None
    air_band_w60s: Optional[DistributionStats] = None
    
    voc_music_proxy_ratio: float

class AnalysisProbabilities(BaseModel):
    model_config = ConfigDict(strict=False)
    
    human: float
    hybrid: float
    ai: float
    verdict_text: str

class AnalysisReport(BaseModel):
    model_config = ConfigDict(strict=False)
    
    metadata: AudioMetadata
    dsp: DSPMetrics
    submithub: SubmitHubMetrics
    feature_vector: Dict[str, float] = Field(default_factory=dict)
    
    spectral_analysis: AnalysisProbabilities
    temporal_analysis: AnalysisProbabilities
    heuristic_combined_analysis: AnalysisProbabilities
    final_probabilities: AnalysisProbabilities
    headline_probabilities: AnalysisProbabilities
    headline_probability_source: str = ""
    
    overall_verdict: str
    model_label: str = ""
    screening_outcome: str
    executive_summary: str = ""
    simple_explanation: str = ""
    main_issue: str = ""
    fix_area: str = ""
    practical_steps: List[str] = Field(default_factory=list)
    analysis_engine: str
    confidence_label: str = "medium"
    confidence_reason: str = ""
    verdict_confidence_label: str = "medium"
    verdict_confidence_reason: str = ""
    ai_verdict_confidence_label: str = "low"
    ai_verdict_confidence_reason: str = ""
    decision_hint: str = ""
    fail_hint: str = ""
    fingerprint_confidence_label: str = "low"
    fingerprint_confidence_reason: str = ""
    fingerprint_level: str = "Low"
    fingerprint_score: int = 0
    fingerprint_archetype: str = "Unknown generator"
    fingerprint_summary: str = ""
    fingerprint_signals: List[str] = Field(default_factory=list)
    fingerprint_score_components: Dict[str, int] = Field(default_factory=dict)
    fingerprint_metrics: Dict[str, float] = Field(default_factory=dict)
    strong_indicator_status: Dict[str, bool] = Field(default_factory=dict)
    weak_indicators: List[str] = Field(default_factory=list)
    production_mimic_indicators: List[str] = Field(default_factory=list)
    
    red_flags: List[str] = Field(default_factory=list)
    strong_red_flags: List[str] = Field(default_factory=list)
    ambiguous_red_flags: List[str] = Field(default_factory=list)
    normal_production_flags: List[str] = Field(default_factory=list)
    green_flags: List[str] = Field(default_factory=list)
    expert_insight: Optional[Dict] = None
    humanization_guide: List[str] = Field(default_factory=list)
    limitations_warnings: List[str] = Field(default_factory=list)
