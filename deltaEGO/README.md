# DeltaEGO: High-Performance Real-time Emotion Engine 

![C++](https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B)
![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg?style=flat&logo=python)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

> **"Bridging the gap between Psychological Theory and High-Performance Computing."**

**DeltaEGO** is a hybrid Emotion AI engine designed for real-time interactive characters. It combines a **custom C++ K-D Tree** for ultra-low latency vector search with a **computational psychology module** that models stress, reward, and emotional lability.

---

## Key Features & Benchmarks

| Component | Feature | Performance Speedup |
| :--- | :--- | :--- |
| **Search Engine** | Iterative K-D Tree (C++) | **~40x Faster** than Python |
| **Latency** | End-to-End Query | **0.0135 ms** (< 15µs) |
| **Architecture** | Python-C++ Hybrid | Zero-Copy via `pybind11` |

## System Modules (Deep Dive)

Click on each module to see the engineering details:

* **[deltaEGO (Orchestration)](./deltaEGO)**
    * The Python layer that manages state, pipelines, and API control.
* **[deltaEGO_VDB (Vector DB)](./deltaEGO_VDB)**
    * **Core Engine.** Why I chose Iterative K-D Tree over HNSW, and how I optimized memory.
* **[deltaEGO_compute (Physics)](./deltaEGO_compute)**
    * **Math Model.** Multithreading strategy for psychological calculation.

---
