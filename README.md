# Machine-Learning-System

# Efficient GPU-Based Retrieval and RAG System for Question Answering

## My Key Contribution

- **Led the design and implementation of an Approximate Nearest Neighbour (ANN) retrieval system** using Inverted File Product Quantisation (IVFPQ), including custom CUDA GPU kernels for distance computations and clustering.
- Delivered significant performance improvements in Top-K retrieval speed and K-means clustering, achieving up to **8000× speedup** over CPU baselines for high-dimensional data.

---

## 1. Why did you start your project?

This project was driven by the need to build scalable and efficient retrieval and serving systems for large-scale question answering. Existing pipelines struggle with performance when processing millions of documents or user queries. Our dual-task system—Approximate Nearest Neighbour retrieval and a Retrieval-Augmented Generation server—was designed to address these limitations using GPU acceleration and system-level optimization techniques. The aim was to make semantic search and document retrieval practical in real-time settings, especially when combined with LLMs.

---

## 2. What issues did you find technically and in a domain context?

### Domain Issues:
- ANN techniques like IVFPQ require complex tuning and are sensitive to clustering quality.
- Retrieval quality often trades off with speed—optimising one risks degrading the other.
- Scaling RAG pipelines typically involves significant infrastructure and GPU memory costs.

### Technical Issues:
- CPU implementations of Top-K retrieval and K-means clustering were prohibitively slow for large datasets.
- CuPy provided acceleration but lacked the full efficiency needed for dense vector operations.
- CUDA memory contention and Out-Of-Memory (OOM) errors emerged during concurrent model inference in the RAG server.
- Load balancing and autoscaling on a shared-GPU machine led to instability under load.

---

## 3. What solutions did you consider?

- **For ANN Retrieval**:
  - CPU-only (NumPy), CuPy-accelerated, and **custom CUDA kernel** implementations for distance functions and clustering.
  - Explored various distance metrics (L2, Cosine, Dot Product, Manhattan) to evaluate tradeoffs in recall vs. latency.
  - Implemented Product Quantisation and residual vector encoding in GPU with tunable parameters (e.g., M, num_probe).

- **For RAG Model Serving**:
  - Queue-based request handling with batcher process.
  - Evaluated various batch sizes (2–32) for optimal tradeoff between latency and throughput.
  - Simulated autoscaling via Python multiprocessing and port-based worker routing.
  - Built a FastAPI-based load balancer using round-robin routing strategy.

---

## 4. What is your final decision among the solutions?

We deployed a robust hybrid system with the following architecture:

### Task 1: Information Retrieval
- **ANN Search Engine using IVFPQ**: GPU-accelerated coarse (K-means) and fine (PQ) clustering.
- **Distance Kernels**: Custom CUDA kernels for Top-K search across multiple similarity metrics.
- **Performance**:
  - Speedup of up to **16×** for L2 distance and **8000×** for high-dimensional K-means vs. CPU.
  - Achieved **60–74% recall** across 1M vector datasets, with query latency below 0.6s.

### Task 2: RAG System
- **Batching + Queueing Server**: Serves queries using background processing for embedding, retrieval, and generation.
- **LLM Backend**: facebook/opt-125m with multilingual-E5 for embeddings.
- **Scalability**: Simulated autoscaler and load balancer to demonstrate elastic response under load.
- **Performance**:
  - Latency reduced from **11.5s → 1.5s**.
  - Throughput increased from **1.3 RPS → 10.6 RPS** at optimal batch sizes.

This project shows that efficient approximate retrieval and inference serving can be achieved on modest hardware with careful system design and GPU optimization. The methods developed here are directly applicable to production-grade systems in semantic search, LLM retrieval augmentation, and real-time recommendation engines.
