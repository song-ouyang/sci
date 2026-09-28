"""
Prompt templates for LLM-based industrial equipment fault diagnosis.

Four strategies:
1. Zero-shot: No examples, just task description
2. Few-shot: Include 2-3 labeled examples
3. Chain-of-Thought (CoT): Step-by-step reasoning
4. Domain-knowledge enhanced: Include engineering knowledge
"""

# ============================================================
# CWRU Bearing Fault Diagnosis Prompts
# ============================================================

CWRU_CLASSES = ["Normal", "IR007", "IR014", "IR021", "B007", "B014", "B021", "OR007", "OR014", "OR021"]
CWRU_CLASSES_COARSE = ["Normal", "Inner Race Fault", "Ball Fault", "Outer Race Fault"]

CWRU_ZERO_SHOT = """You are an industrial equipment fault diagnosis expert. Based on the following vibration signal features from a rolling element bearing, classify the bearing condition.

Possible fault types: {classes}

{sensor_data}

Respond with ONLY the fault type label, nothing else."""

CWRU_FEW_SHOT = """You are an industrial equipment fault diagnosis expert. Based on vibration signal features from a rolling element bearing, classify the bearing condition including fault severity.

Possible fault types: {classes}

Here are some examples:

Example 1 (Normal):
  Time-domain: RMS=0.0523g, Kurtosis=-0.15, Crest factor=3.12
  Frequency-domain: Spectral centroid=780Hz, Total power=0.002
  Diagnosis: Normal

Example 2 (IR007 - minor inner race fault):
  Time-domain: RMS=0.1534g, Kurtosis=3.42, Crest factor=5.21
  Frequency-domain: Spectral centroid=2450Hz, Total power=0.012
  Diagnosis: IR007

Example 3 (IR021 - severe inner race fault):
  Time-domain: RMS=0.2814g, Kurtosis=22.5, Crest factor=10.2
  Frequency-domain: Spectral centroid=3100Hz, Total power=0.042
  Diagnosis: IR021

Example 4 (OR014 - moderate outer race fault):
  Time-domain: RMS=0.3102g, Kurtosis=7.23, Crest factor=5.34
  Frequency-domain: Spectral centroid=2890Hz, Total power=0.048
  Diagnosis: OR014

Now diagnose the following:

{sensor_data}

Respond with ONLY the fault type label, nothing else."""

CWRU_COT = """You are an industrial equipment fault diagnosis expert. Analyze the following vibration signal features from a rolling element bearing step by step.

Possible fault types: {classes}

{sensor_data}

Please analyze step by step:
1. Examine time-domain features (RMS, kurtosis, crest factor) - are they elevated? How elevated?
2. Examine frequency-domain features - do dominant frequencies correspond to BPFI (inner race), BPFO (outer race), or BSF (ball) harmonics?
3. Determine fault TYPE based on which characteristic frequency is excited.
4. Determine fault SEVERITY based on amplitude levels: 007 (minor, lower RMS/kurtosis), 014 (moderate), 021 (severe, highest RMS/kurtosis).
5. State your final diagnosis combining type and severity.

After your analysis, on the LAST line write ONLY: Diagnosis: <fault_type>"""

CWRU_DOMAIN = """You are an industrial equipment fault diagnosis expert with deep knowledge of rolling element bearing dynamics.

Engineering context:
- Bearing type: 6205-2RS JEM SKF deep groove ball bearing
- Characteristic frequencies at ~1797 RPM: BPFO≈107.4Hz, BPFI≈162.2Hz, BSF≈141.2Hz, FTF≈11.2Hz
- Inner race faults produce impulses at BPFI with amplitude modulation at shaft frequency
- Outer race faults produce impulses at BPFO, typically strongest at the load zone
- Ball faults produce impulses at 2×BSF, often with random amplitude variation
- Normal bearings show low kurtosis (<4), low RMS, no dominant fault frequencies

Fault severity indicators (007=0.007 inch, 014=0.014 inch, 021=0.021 inch diameter):
- 007 (minor): RMS typically 0.05-0.20g, moderate kurtosis increase, subtle frequency shifts
- 014 (moderate): RMS typically 0.15-0.40g, clear kurtosis elevation, distinct fault frequency peaks
- 021 (severe): RMS typically 0.25-0.70g, very high kurtosis (>15 for IR), strong broadband excitation
- Severity progression: spectral centroid shifts higher, total spectral power increases, crest factor rises with severity
- Inner race faults show the most dramatic kurtosis increase with severity
- Outer race faults show the most dramatic RMS increase with severity
- Ball faults show intermediate patterns with more random amplitude variation

Possible fault types: {classes}

{sensor_data}

Based on your engineering expertise and the sensor data above, what is the fault type and severity?
Respond with ONLY the fault type label, nothing else."""


# ============================================================
# C-MAPSS Engine Health Prompts
# ============================================================

CMAPSS_CLASSES = ["Healthy", "Degrading", "Critical"]

CMAPSS_ZERO_SHOT = """You are a turbofan engine health monitoring expert. Based on the following sensor trend data, classify the engine's health stage.

Health stages:
- Healthy: Engine operating within normal parameters, no significant degradation
- Degrading: Engine showing signs of wear, performance declining
- Critical: Engine nearing end of useful life, immediate attention needed

{sensor_data}

Respond with ONLY the health stage (Healthy, Degrading, or Critical), nothing else."""

CMAPSS_FEW_SHOT = """You are a turbofan engine health monitoring expert. Classify the engine's health stage based on sensor trends.

Health stages: Healthy, Degrading, Critical

Example 1 (Healthy):
  Sensor trends: All sensors stable, minimal deviation from baseline
  HPC outlet temperature trend: +0.0002/cycle (stable)
  Physical core speed trend: -0.0001/cycle (stable)
  Diagnosis: Healthy

Example 2 (Degrading):
  Sensor trends: Several sensors showing gradual drift
  HPC outlet temperature trend: +0.0045/cycle (increasing)
  Physical core speed trend: -0.0038/cycle (decreasing)
  Diagnosis: Degrading

Example 3 (Critical):
  Sensor trends: Multiple sensors significantly deviated
  HPC outlet temperature trend: +0.0120/cycle (rapidly increasing)
  Physical core speed trend: -0.0095/cycle (rapidly decreasing)
  Diagnosis: Critical

Now diagnose:

{sensor_data}

Respond with ONLY the health stage, nothing else."""

CMAPSS_COT = """You are a turbofan engine health monitoring expert. Analyze the following sensor trends step by step.

Health stages: Healthy, Degrading, Critical

{sensor_data}

Please analyze step by step:
1. Check temperature sensors (T24, T30, T50) - are they trending upward? This indicates degradation.
2. Check pressure sensors (P30, Ps30) - declining pressure suggests component wear.
3. Check speed sensors (Nf, Nc) - deviations from expected values indicate efficiency loss.
4. Check fuel-related sensors (phi, farB) - increasing fuel ratio suggests compensating for degradation.
5. Assess overall severity based on number and magnitude of deviating sensors.

After your analysis, on the LAST line write ONLY: Health stage: <stage>"""

CMAPSS_DOMAIN = """You are a turbofan engine health monitoring expert with deep knowledge of gas turbine degradation physics.

Engineering context:
- C-MAPSS simulates HPC (High Pressure Compressor) degradation and fan degradation
- Key degradation indicators: T30 (HPC outlet temp) increases as compressor efficiency drops
- Nc (core speed) must increase to maintain thrust as efficiency degrades
- P30 (HPC outlet pressure) decreases with compressor blade erosion
- phi (fuel flow ratio) increases as engine compensates for efficiency loss
- BPR (bypass ratio) shifts as fan performance degrades
- Sensor trend slope magnitude indicates degradation rate
- Trends >0.005/cycle in multiple sensors typically indicate advanced degradation

Health stages: Healthy (>80 cycles remaining), Degrading (30-80 cycles), Critical (<30 cycles)

{sensor_data}

Based on your expertise, classify the health stage.
Respond with ONLY: Healthy, Degrading, or Critical."""


# ============================================================
# TEP Chemical Process Prompts
# ============================================================

TEP_ZERO_SHOT = """You are a chemical process monitoring expert. Based on the following process variable readings from a Tennessee Eastman chemical plant, determine if the process is operating normally or if a fault is present.

Possible conditions: {classes}

{sensor_data}

Respond with ONLY the condition label, nothing else."""

TEP_FEW_SHOT = """You are a chemical process monitoring expert. Classify the operating condition of a Tennessee Eastman process plant.

Possible conditions: {classes}

Example 1 (Normal):
  Reactor pressure: 2705 kPa, Reactor temperature: 120.4°C, Reactor level: 65.0%
  All variables within normal operating ranges.
  Diagnosis: Normal

Example 2 (Fault_1 - A/C feed ratio change):
  Reactor pressure: 2705 kPa, Reactor temperature: 120.8°C, Reactor level: 63.2%
  A feed (stream 1) shows significant deviation, composition shift detected.
  Diagnosis: Fault_1

Example 3 (Fault_6 - A feed loss):
  Reactor pressure: 2680 kPa, Reactor temperature: 119.1°C, Reactor level: 58.4%
  A feed dropped dramatically, reactor level declining.
  Diagnosis: Fault_6

Now diagnose:

{sensor_data}

Respond with ONLY the condition label, nothing else."""

TEP_COT = """You are a chemical process monitoring expert. Analyze the Tennessee Eastman process readings step by step.

Possible conditions: {classes}

{sensor_data}

Please analyze step by step:
1. Check reactor conditions (pressure, temperature, level) - are they within normal ranges?
2. Check feed streams - any sudden changes in flow rates or compositions?
3. Check product quality indicators - any composition shifts in output streams?
4. Check manipulated variables - are control valves at unusual positions?
5. Identify the pattern - does it match a known fault signature?

After your analysis, on the LAST line write ONLY: Condition: <label>"""

TEP_DOMAIN = """You are a chemical process monitoring expert with deep knowledge of the Tennessee Eastman process.

Engineering context:
- The TEP is a realistic simulation of an industrial chemical process with reactor, separator, and stripper
- Normal operation: reactor pressure ~2705 kPa, temperature ~120.4°C, level ~65%
- IDV(1): A/C feed ratio step change - affects stream 4 composition, reactor level shifts
- IDV(2): B composition step change - product quality affected
- IDV(4): Reactor cooling water temperature step - reactor temperature rises
- IDV(6): A feed loss - dramatic drop in stream 1 flow, reactor level drops
- IDV(7): C header pressure loss - reduced feed availability
- IDV(13): Reaction kinetics slow drift - gradual performance change
- Key diagnostic strategy: compare current readings against normal baselines

Possible conditions: {classes}

{sensor_data}

Based on your expertise, classify the condition.
Respond with ONLY the condition label, nothing else."""


# ============================================================
# Unified access
# ============================================================

PROMPT_REGISTRY = {
    "cwru": {
        "zero_shot": CWRU_ZERO_SHOT,
        "few_shot": CWRU_FEW_SHOT,
        "cot": CWRU_COT,
        "domain": CWRU_DOMAIN,
        "classes": CWRU_CLASSES,
    },
    "cmapss": {
        "zero_shot": CMAPSS_ZERO_SHOT,
        "few_shot": CMAPSS_FEW_SHOT,
        "cot": CMAPSS_COT,
        "domain": CMAPSS_DOMAIN,
        "classes": CMAPSS_CLASSES,
    },
    "tep": {
        "zero_shot": TEP_ZERO_SHOT,
        "few_shot": TEP_FEW_SHOT,
        "cot": TEP_COT,
        "domain": TEP_DOMAIN,
        "classes": None,  # set dynamically based on faults used
    },
}


def get_prompt(dataset: str, strategy: str, sensor_data: str,
               classes: list[str] | None = None) -> str:
    """Build a complete prompt for a given dataset and strategy."""
    registry = PROMPT_REGISTRY[dataset]
    template = registry[strategy]
    cls = classes or registry["classes"]
    class_str = ", ".join(cls) if cls else ""

    return template.format(
        sensor_data=sensor_data,
        classes=class_str,
    )
