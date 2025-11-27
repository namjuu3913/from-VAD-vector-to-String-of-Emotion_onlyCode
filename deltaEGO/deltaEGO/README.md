## 🧩 Python orchestration layer (`deltaEGO` class)

The `deltaEGO` Python class is a **high-level wrapper** around:

- the C++ core (`deltaEGO_compute`), and
- the VAD-based vector database (`EGOSearcher` from `deltaEGO_VDB`).

It keeps track of:

- the latest emotion,
- full VAD history,
- analysis history (metrics from the C++ engine),

and exposes a clean API that an agent (e.g. **Fuli**) can call from Python.

```python
from deltaEGO import deltaEGO, VAD_search
from pathlib import Path
```
## Core idea

**1. ```VADsearch()```**

  * Takes a VAD point (V, A, D) with search parameters.
  * Calls the VDB (EGOSearcher.search) to find similar emotional points.
  * Updates:
    * ```last_emotion```
    * ```last_emotion_VADPoint```
    * ```emotion_history```
    * ```emotion_history_VADPoint```
  * Optionally triggers automatic analysis.

**2. ```analize_VAD()```**

  * Builds a ```compute_in``` bundle from:
    * current VAD point
    * full history
    * optional ```prev``` point
    * optional custom ```weights```, ```variables```, ```emotion_base```
  * Converts this Python ```TypedDict``` into the C++ ```compute_in``` struct.
  * Calls ```deltaEGO_compute.compute(...)``` (C++ engine).
  * Appends a Python-friendly version of the result into ```analysis_history```.

The agent layer (e.g. Fuli) only needs to:
  * call ```VADsearch()``` whenever a new VAD vector is available,
  * optionally call ```analize_VAD()``` to get detailed metrics and update internal state.
---
## Type overview

All types are defined as TypedDict for clarity and static checking:

  * Search input
```python
class VAD_search(TypedDict):
    V: float          # valence
    A: float          # arousal
    D: float          # dominance
    k: int            # k-NN neighbors
    dis: float        # search distance / radius
    sigma: Optional[float]
    api: Optional[str]
```

* Emotion point (Python-side)
```python
class VADPoint(TypedDict):
    v: float
    a: float
    d: float
    timestamp: float
    owner: str
```

* Analysis input bundle
```python
class compute_in(TypedDict, total=False):
    current: VADPoint
    history: List[VADPoint]
    prev: Optional[VADPoint]
    emotion_base: Optional[EGO_axis]
    variables: Optional[variable]
    weights: Optional[weight]
```

* Analysis result (Python-side)
```python
class InstantMetrics(TypedDict):
    stress: float
    reward: float
    ratio_total: float
    stress_ratio: float
    reward_ratio: float
    deviation: float

class DynamicMetrics(TypedDict):
    delta: VADPoint
    affective_lability: float

class CumulativeMetrics(TypedDict):
    average_area: VAD_ave
    stress: float
    reward: float
    total: float
    stress_ratio: float
    reward_ratio: float

class AnalysisResult_py(TypedDict):
    instant: InstantMetrics
    dynamics: DynamicMetrics
    cumulative: CumulativeMetrics
```

Internally, the class also works with C++ types bound via ```pybind11```:
```python
CppVADPoint      = deltaEGO_compute.VADPoint
CppEGO_axis      = deltaEGO_compute.EGO_axis
CppVariable      = deltaEGO_compute.variable
CppWeight        = deltaEGO_compute.weight
CppComputeIn     = deltaEGO_compute.compute_in
CppAnalysisResult = deltaEGO_compute.AnalysisResult
```
## ```deltaEGO``` class
```python
class deltaEGO:
    def __init__(self, character_name: str, save_path: Optional[Path] = None):
        self.ego_character = character_name
        self.EGO_save_path = save_path  # TODO: future persistence

        # state
        self.last_emotion = None
        self.last_emotion_VADPoint = None
        self.emotion_history = []
        self.emotion_history_VADPoint = []
        self.analysis_history = []

        # C++ modules
        self.ego_searcher = EGOSearcher()
        self.compute = deltaEGO_compute.compute

        # defaults
        self.DEFAULT_SIGMA = 0.5
        self.DEFAULT_API_OPT = "knn~l2 -S"

        self.default_axis = EGO_axis(
            baseline=VADPoint(v=0.0, a=0.0, d=0.0, timestamp=0.0, owner="base"),
            stabilityRadius=0.3,
        )
        self.default_weights = weight()     # use C++ defaults
        self.default_variables = variable() # use C++ defaults

        self.automatic_analize = False
        self.save = False  # TODO
```
---
## VADsearch(...)
```python
def VADsearch(self, in_VAD: VAD_search) -> dict:
    api_opt = in_VAD.get("api") or self.DEFAULT_API_OPT
    sigma = in_VAD.get("sigma") or self.DEFAULT_SIGMA

    ego_result = self.ego_searcher.search(
        V=in_VAD["V"],
        A=in_VAD["A"],
        D=in_VAD["D"],
        k=in_VAD["k"],
        d=in_VAD["dis"],
        SIGMA=sigma,
        opt=api_opt,
    )

    current_vad_point = VADPoint(
        v=in_VAD["V"],
        a=in_VAD["A"],
        d=in_VAD["D"],
        timestamp=time.time(),
        owner=self.ego_character,
    )

    self.emotion_history.append(ego_result)
    self.emotion_history_VADPoint.append(current_vad_point)

    self.last_emotion = ego_result
    self.last_emotion_VADPoint = current_vad_point

    if self.automatic_analize:
        self.analize_VAD(return_analysis=False)

    return ego_result
```
* Input: one VAD point + search params (```k```, ```dis```, etc.)
* Output: raw search result from ```EGOSearcher``` (Python ```dict```)
* Side effects:
    * appends to ```emotion_history``` / ```emotion_history_VADPoint```
    * updates ```last_emotion``` / ```last_emotion_VADPoint```
* optional auto-call to ```analize_VAD()```
---
## ```analize_VAD(...)```
```python
def analize_VAD(
    self,
    weights: Optional[weight] = None,
    variables: Optional[variable] = None,
    emotion_base: Optional[EGO_axis] = None,
    return_analysis: bool = True,
    append_emotion: bool = True,
) -> Optional[CppAnalysisResult]:
    if not self.last_emotion_VADPoint:
        warnings.warn("Warning: No VAD data to analyze. Call VADsearch() first.", RuntimeWarning)
        return None

    prev_point_dict = None
    if len(self.emotion_history_VADPoint) > 1:
        prev_point_dict = self.emotion_history_VADPoint[-2]

    input_bundle_dict = compute_in(
        current=self.last_emotion_VADPoint,
        history=self.emotion_history_VADPoint,
        prev=prev_point_dict,
        emotion_base=emotion_base or self.default_axis,
        variables=variables or self.default_variables,
        weights=weights or self.default_weights,
    )

    input_bundle_cpp = self._build_cpp_input_bundle(input_bundle_dict)
    analyzed_result: CppAnalysisResult = self.compute(input_bundle_cpp)

    if append_emotion:
        self.analysis_history.append(self.analysis_cpp_to_py(analyzed_result))

    if return_analysis:
        return analyzed_result
    else:
        return None
```
* Requires at least one VAD point (```VADsearch()``` must be called first).
* Builds a ```compute_in``` bundle from:
  * current emotion
  * full history
  * previous emotion (if exists)
  * axis/weights/variables (defaults or overrides)
* Converts this bundle into C++ structs and calls the C++ engine.
* Optionally appends a Python-friendly version into ```analysis_history```.
* Returns the raw ```CppAnalysisResult``` if ```return_analysis=True```.
---

## Example: minimal usage from an agent
```python
from deltaEGO_wrapper import deltaEGO, VAD_search  # adjust import to your layout

ego = deltaEGO(character_name="Fuli")

# 1) a new VAD vector comes from an LLM or other model:
vad_input: VAD_search = {
    "V": 0.1,
    "A": 0.7,
    "D": -0.2,
    "k": 8,
    "dis": 0.5,
    # "sigma": 0.5,  # optional
    # "api": "knn~l2 -S",  # optional
}

search_result = ego.VADsearch(vad_input)

# 2) run analysis on the updated history:
analysis_cpp = ego.analize_VAD()
analysis_py = ego.analysis_cpp_to_py(analysis_cpp)

print("Instant stress:", analysis_py["instant"]["stress"])
print("Cumulative reward:", analysis_py["cumulative"]["reward"])
```

In practice, your agent (Fuli) would:

* call ```VADsearch(...)``` after each user/agent turn,

* call ```analize_VAD(...)``` when it needs updated emotional metrics,

* use ```analysis_history```, ```last_emotion_VADPoint```, etc. to drive memory and behavior.
