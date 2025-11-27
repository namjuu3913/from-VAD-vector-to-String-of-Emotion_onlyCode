## 📚 VAD Vector Database (custom KD-Tree)

deltaEGO uses a custom **KD-Tree–based vector database** to store and query emotions
in VAD (Valence–Arousal–Dominance) space.  
Each entry corresponds to one emotion term with its VAD coordinates.

Internally this is implemented in C++ as `KDTree` (in `VAD_customVDB.*`). :contentReference[oaicite:0]{index=0}

---

### Data format & loading

The VDB is populated from a JSON file of the following shape:

```jsonc
[
  {
    "term": "cheer",
    "valence": 0.8,
    "arousal": 0.6,
    "dominance": 0.5
  },
  ...
]
```
The loader:
1. parses the JSON array.
2. For each element creates an ```Emotion```:
    * ```term``` – emotion label (e.g. ```"cheer"```)
    * ```point``` – 3D coordinates ```(x = valence, y = arousal, z = dominance)```
3. Stores them in ```Emotions``` (a contiguous ```std::vector<Emotion>```).
4. Builds a KD-Tree over the indices of ```Emotions```, using an iterative algorithm.
5. Computes per-axis standard deviation (```AxisScale```) for later whitened Gaussian similarity (see below).
---
## Iterative KD-Tree build

Instead of a recursive build (which risks stack overflows and reallocations), the KD-Tree is built iteratively using an explicit stack of frames:
```cpp
struct Frame {
    int l, r;       // half-open range [l, r) in the index buffer
    int depth;      // determines split axis: depth % 3
    int parent;     // parent node index (-1 for root)
    bool is_left;   // whether this node is the left child of its parent
};
```
Algorithm sketch:

1. Start with frame ```{l=0, r=N, depth=0, parent=-1}```.

2. For each frame, choose split axis = ```depth % 3```.

3. Use ```std::nth_element``` on the index buffer to place the median at position ```median```.

4. Create a ```Node``` with:

    * ```idx``` = ```P_buffer[median]``` (index into ```Emotions```)
    * ```axis``` = split axis
    * ```left / right``` = initially ```-1```

5. Link this node to its parent (or mark it as ```root``` if ```parent``` = ```-1```).
6. Push right and left subranges onto the stack with ```depth + 1```.

This gives a compact KD-Tree where:

  * nodes are stored in a single std::vector<Node>,
  * each node points into the emotion array instead of duplicating data.
