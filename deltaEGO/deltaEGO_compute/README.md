## 🧮 Emotion metrics core (`EGO_compute`)

This module takes a **history of VAD points** and computes:

- instant stress / reward (current moment),
- affective lability (emotional whiplash),
- deviation from baseline,
- cumulative stress / reward over time,
- average “emotion area” in VAD space,

and returns everything packed into an `AnalysisResult` struct that deltaEGO exposes
to Python. :contentReference[oaicite:0]{index=0}

Input type (conceptual):

```cpp
struct compute_in {
    VADPoint                        current;      // latest VAD
    std::vector<VADPoint>           history;      // full history (time-ordered)
    std::optional<VADPoint>         prev;         // previous point (if any)
    std::optional<EGO_axis>         emotion_base; // baseline + stability radius
    std::optional<weight>           weights;      // stress/reward weights
    std::optional<variable>         variables;    // theta_0, dampening_factor, ...
};
```
Output type (conceptual):
```cpp
struct AnalysisResult {
    InstantMetrics   instant;
    DynamicMetrics   dynamics;
    CumulativeMetrics cumulative;
};
```
---
## Core helpers (O(1))
All low-level metrics are built from small O(1) functions: 
```cpp
// Euclidean distance in VAD space
double get_distance(const VADPoint& a, const VADPoint& b);

// Numerically stable sigmoid
double sigmoid(double x);

// Pack stress/reward + their ratios
Ratio get_stress_reward_ratio(double stress, double reward);

// Temporal delta of VAD (per second)
VADPoint calculate_delta(const VADPoint& prev, const VADPoint& current);

// Instant stress: depends on valence, arousal, and distance from baseline
double calculate_instant_stress(
    const VADPoint& current,
    const VADPoint& baseline,
    double stabilityRadius,
    double weightA_stress,
    double weightV_stress,
    double dampening_factor);

// Affective lability (emotional whiplash), based on the delta direction
double calculate_affective_lability(
    const VADPoint& delta,
    double weight_k,
    double theta_0);

// Reward index (dopamine-like), based on high valence & arousal
double calculate_reward_index(
    const VADPoint& current,
    double weightV_reward,
    double weightA_reward);
```
These are bundled in `get_O1_functions_async(...)`, which returns:
  * delta (VAD velocity),
  * instant stress / reward,
  * stress & reward ratios,
  * affective lability,
  * deviation from baseline (distance).
---
## History-based metrics (O(n))
For long-term behavior, the engine walks over the entire history:
```cpp
// Average center + radius of the emotion cloud in VAD space
VAD_ave calculate_average(const std::vector<VADPoint>& history);

// Time-integrated stress (anti-derivative over time)
double calculate_cumulative_stress(
    const std::vector<VADPoint>& history,
    const VADPoint& baseline,
    double stabilityRadius,
    double weightA_stress,
    double weightV_stress,
    double dampening_factor);

// Time-integrated reward
double calculate_cumulative_reward(
    const std::vector<VADPoint>& history,
    double weightV_reward,
    double weightA_reward);
```
These are wrapped in `get_Tn_functions_async(...)`, which returns a `Ratio` with:
  * raw cumulative stress / reward,
  * their total,
  * and normalized ratios.
    
---
## Multithreaded execution (`EGO_compute`)
The main entry point is:
```cpp
AnalysisResult EGO_compute(const compute_in& user_in);
```
It runs three tasks in parallel using `std::async`:
```cpp
auto thread_average = std::async(
    std::launch::async,
    calculate_average,
    std::ref(user_in.history)
);

auto thread_Tn_funcs = std::async(
    std::launch::async,
    get_Tn_functions_async,
    std::ref(user_in.history),
    std::ref(base.baseline),
    base.stabilityRadius,
    w.weightA_stress,
    w.weightV_stress,
    v.dampening_factor,
    w.weightV_reward,
    w.weightA_reward
);

auto thread_O1_funcs = std::async(
    std::launch::async,
    get_O1_functions_async,
    std::ref(user_in.prev),
    std::ref(user_in.current),
    std::ref(base.baseline),
    base.stabilityRadius,
    w.weightA_stress,
    w.weightV_stress,
    w.weightV_reward,
    w.weightA_reward,
    v.dampening_factor,
    w.weight_k,
    v.theta_0
);
```
  * `calculate_average(...)`
    
    → average VAD center + radius (cumulative “emotion cloud”)
  * `get_Tn_functions_async(...)`
    
    → cumulative stress & reward (time-integrated)
  * `get_O1_functions_async(...)`
    
    → instant metrics (stress, reward, whiplash, deviation, ratios)

Results are collected and packed into `AnalysisResult`:
```cpp
AnalysisResult final_result;

// InstantMetrics
final_result.instant.stress        = o1_results.instant_stress;
final_result.instant.reward        = o1_results.instant_reward;
final_result.instant.ratio_total   = o1_results.instant_ratio_total;
final_result.instant.stress_ratio  = o1_results.instant_stress_ratio;
final_result.instant.reward_ratio  = o1_results.instant_reward_ratio;
final_result.instant.deviation     = o1_results.deviation;

// DynamicMetrics
final_result.dynamics.delta              = o1_results.delta;
final_result.dynamics.affective_lability = o1_results.affective_lability;

// CumulativeMetrics
final_result.cumulative.average_area = average_result;
final_result.cumulative.stress       = n_results.stress_raw;
final_result.cumulative.reward       = n_results.reward_raw;
final_result.cumulative.total        = n_results.ratio_total;
final_result.cumulative.stress_ratio = n_results.stress_ratio;
final_result.cumulative.reward_ratio = n_results.reward_ratio;
```
---
## How this is used by Python (`deltaEGO`)

On the Python side, the `deltaEGO` wrapper:
1. Builds a compute_in bundle from:
    * `current` VAD,
    * full `history`,
    * optional `prev`,
    * `emotion_base` (baseline & stability radius),
    * `weights` / `variables` (hyperparameters).

2. Converts it into the C++ struct using `pybind11`.
3. Calls `EGO_compute(...)`.
4. Stores the resulting `AnalysisResult` both as:
    * the raw C++ object, and
    * a Python `TypedDict` (`AnalysisResult_py`) for easy use in agents.

From an agent’s perspective, this module answers:

<i>“Given everything this character has felt so far, how stressed/rewarded are they now ,how volatile is their emotion, and what does their long-term emotional field look like?”


