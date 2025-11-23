# Benchmark code by Gemini 3

import time
import json
import random
import numpy as np
import os
import sys

# ---------------------------------------------------------
# [Setup] Ensure current directory is in sys.path
# This fixes "ModuleNotFoundError" when running from terminal
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# ---------------------------------------------------------
# [Config] Path to the original data file
# ---------------------------------------------------------
# Assuming directory structure:
# Project/
#   ├── Distilled_data/final.json
#   └── deltaEGO/benchmark.py
TARGET_FILE = os.path.join(current_dir, "../Distilled_data/final.json")

# ---------------------------------------------------------
# 1. Import DeltaEGO (Your Module)
# ---------------------------------------------------------
try:
    from deltaEGO import deltaEGO
except ImportError as e:
    print(f"❌ Error: Import failed. Details: {e}")
    print("👉 Please ensure 'deltaEGO.py' and compiled C++ files (.so/.pyd) are in this folder.")
    sys.exit(1)

# ---------------------------------------------------------
# 2. Import Faiss (Industry Standard)
# ---------------------------------------------------------
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    print("\n⚠️  [Warning] Faiss is not installed. (pip install faiss-cpu)")
    print("👉 Skipping Faiss test. Comparing Python vs DeltaEGO only.\n")
    HAS_FAISS = False

# ---------------------------------------------------------
# 3. Helper: Load Data
# ---------------------------------------------------------
def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"❌ Error: Dataset file not found at: {os.path.abspath(filepath)}")
        print("👉 Please check the 'TARGET_FILE' path in the script.")
        sys.exit(1)

    print(f"📂 Loading dataset: '{filepath}'...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # For Pure Python (List of Dictionaries)
    py_data = data

    # For Faiss (Numpy float32 array: [[v, a, d], ...])
    vectors = [
        [d['valence'], d['arousal'], d['dominance']] 
        for d in data
    ]
    np_data = np.array(vectors).astype('float32')

    print(f"✅ Data Loaded: {len(py_data)} items")
    return py_data, np_data

# ---------------------------------------------------------
# 4. Search Algorithms
# ---------------------------------------------------------

# [Contender 1] Pure Python (Linear Scan, O(N))
def python_search(query, dataset):
    qv, qa, qd = query
    best_dist = float('inf')
    
    # Linear scan: Calculate distance for every item
    for item in dataset:
        # Euclidean distance squared (sqrt omitted for speed)
        dist = (qv - item['valence'])**2 + \
               (qa - item['arousal'])**2 + \
               (qd - item['dominance'])**2
        
        if dist < best_dist:
            best_dist = dist
    return best_dist

# [Contender 2] Faiss (FlatL2, Brute-Force with SIMD optimization)
def setup_faiss(np_vectors):
    dimension = 3
    index = faiss.IndexFlatL2(dimension) # Brute-Force L2 Search
    index.add(np_vectors)
    return index

# ---------------------------------------------------------
# 5. Main Benchmark Execution
# ---------------------------------------------------------
def run_benchmark():
    print("\n" + "="*60)
    print("⚔️  [BENCHMARK] Python vs Faiss vs DeltaEGO (C++)  ⚔️")
    print("="*60)

    # 1. Prepare Data
    py_data, np_data = load_data(TARGET_FILE)

    # 2. Initialize DeltaEGO (C++)
    print("⚙️  Initializing DeltaEGO...", end=" ")
    my_engine = deltaEGO(character_name="BenchmarkBot")

    # 3. Initialize Faiss
    faiss_index = None
    if HAS_FAISS:
        print("⚙️  Building Faiss Index...", end=" ")
        faiss_index = setup_faiss(np_data)
        print("Done.")

    # 4. Generate Random Queries
    ITERATIONS = 1000
    # Random vectors in range [-1, 1]
    queries = np.random.rand(ITERATIONS, 3).astype('float32') * 2 - 1 
    
    print(f"\n🚀 Starting Benchmark ({ITERATIONS} random queries)...")

    # --- [Round 1] Pure Python ---
    print("   Running Python...", end="\r")
    start = time.perf_counter()
    for i in range(ITERATIONS):
        python_search(queries[i], py_data)
    time_py = (time.perf_counter() - start) / ITERATIONS * 1000 # ms
    print(f"✅ Python Finished      ")

    # --- [Round 2] Faiss ---
    time_faiss = 0
    if HAS_FAISS:
        print("   Running Faiss...", end="\r")
        start = time.perf_counter()
        # Loop individually to measure single-query latency (Fair comparison)
        for i in range(ITERATIONS):
            faiss_index.search(queries[i].reshape(1, -1), 1)
        time_faiss = (time.perf_counter() - start) / ITERATIONS * 1000 # ms
        print(f"✅ Faiss Finished       ")

    # --- [Round 3] DeltaEGO ---
    print("   Running DeltaEGO...", end="\r")
    start = time.perf_counter()
    for i in range(ITERATIONS):
        my_engine.VADsearch({
            'V': float(queries[i][0]), 
            'A': float(queries[i][1]), 
            'D': float(queries[i][2]),
            'k': 1, 'dis': 10.0, 'api': 'knn' 
        })
    time_my = (time.perf_counter() - start) / ITERATIONS * 1000 # ms
    print(f"✅ DeltaEGO Finished    ")

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print(f"📊  FINAL RESULTS (Average Latency per Query)")
    print("-" * 60)
    print(f"1. 🐍 Python (Linear):  {time_py:.4f} ms")
    if HAS_FAISS:
        print(f"2. 🔵 Faiss  (FlatL2):  {time_faiss:.4f} ms")
    print(f"3. 🟢 DeltaEGO (KDTree): {time_my:.4f} ms")
    
    print("-" * 60)
    print("📢  [Conclusion]")
    
    # Speedup vs Python
    speedup_py = time_py / time_my if time_my > 0 else 0
    print(f"  - Approx. [{speedup_py:.1f}x] faster than Pure Python.")

    if HAS_FAISS:
        if time_my < time_faiss:
            print(f"  - 🏆 Faster than Faiss by [{time_faiss/time_my:.1f}x]! (Tree structure advantage on small datasets)")
        else:
            print(f"  - Slower than Faiss by [{time_my/time_faiss:.1f}x].")
            print("    (Note: Faiss uses AVX/SIMD optimizations for brute-force,")
            print("           while DeltaEGO uses K-D Tree to reduce search space.)")

if __name__ == "__main__":
    run_benchmark()