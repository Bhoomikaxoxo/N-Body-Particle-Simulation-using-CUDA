# 🌌 2D N-Body Gravity Simulation & CUDA Pipeline Workflow

An educational, high-performance case study demonstrating **all-pairs gravitational $N$-body simulation** and the **canonical 5-stage CUDA host pipeline workflow**. This project pairs an optimized CUDA C++ implementation with an interactive HTML5 Canvas visualizer.

---

## 📌 Overview

In an $N$-body gravitational simulation, every particle experiences the net gravitational pull of every other particle in the system. For $N$ particles, this requires calculating $N \times (N - 1)$ pairwise interactions per timestep—an $\mathcal{O}(N^2)$ computational complexity.

This repository demonstrates how to accelerate this workload using NVIDIA GPUs via CUDA, focusing on:
1. **The Standard 5-Step CUDA Host Workflow** (Allocate $\to$ Copy In $\to$ Launch $\to$ Copy Out $\to$ Free).
2. **Shared-Memory Tiling** to dramatically cut global memory bandwidth demands.
3. **Interactive Simulation & Visualization** via an in-browser HTML5 Canvas app with live controls and performance metrics.

---

## 📂 Project Structure

```
.
├── nbody_cuda.cu              # CUDA C++ source code (kernel + host pipeline)
├── nbody_cuda_simulation.html # Interactive 2D Canvas simulator & educational UI
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

### File Breakdown

- **[`nbody_cuda.cu`](./nbody_cuda.cu)**:
  - Implements the complete CUDA host lifecycle and the `nbody_step` kernel.
  - Features 1D grid/block decomposition (`BLOCK_SIZE = 256`), shared-memory staging tiles (`__shared__ float4 source_tile[256]`), softened gravity, and boundary checks.
- **[`nbody_cuda_simulation.html`](./nbody_cuda_simulation.html)**:
  - Self-contained web app with an animated 2D particle field.
  - Interactive controls (particle count, simulation speed, pause/resume, reset).
  - Visual breakdown of the 5 host steps and GPU architecture concepts.

---

## ⚙️ The Canonical 5-Step CUDA Pipeline

Both the CUDA implementation and the interactive guide follow the standard host-side GPU execution lifecycle:

```mermaid
flowchart LR
    S1["1. ALLOCATE\nmalloc (Host)\ncudaMalloc (Device)"] --> S2["2. COPY IN\ncudaMemcpy\nHost -> Device"]
    S2 --> S3["3. LAUNCH\n<<<grid, block>>>\nnbody_step kernel"]
    S3 --> S4["4. COPY OUT\ncudaMemcpy\nDevice -> Host"]
    S4 --> S5["5. FREE\nfree (Host)\ncudaFree (Device)"]
```

| Step | Action | Functions / API | Description |
|:---|:---|:---|:---|
| **01** | **Allocate** | `std::malloc`, `cudaMalloc` | Allocates memory for positions and velocities on both CPU host and GPU device. |
| **02** | **Copy In** | `cudaMemcpy(..., cudaMemcpyHostToDevice)` | Transfers initial particle state vectors from host RAM to GPU VRAM. |
| **03** | **Launch** | `nbody_step<<<blocks, BLOCK_SIZE>>>(...)` | Dispatches the GPU grid across Streaming Multiprocessors (SMs) to update positions and velocities. |
| **04** | **Copy Out** | `cudaMemcpy(..., cudaMemcpyDeviceToHost)` | Transfers computed next-step positions and velocities back to host memory. |
| **05** | **Free** | `std::free`, `cudaFree` | Frees all dynamically allocated host and device resources to prevent memory leaks. |

---

## 🔬 Physics & Kernel Implementation Details

### 1. Mathematical Formulation

The gravitational acceleration $\mathbf{a}_i$ on particle $i$ due to all source particles $j$ is computed as:

$$\mathbf{a}_i = \sum_{j=0, j \neq i}^{N-1} G \cdot m_j \frac{\mathbf{r}_j - \mathbf{r}_i}{\left(\|\mathbf{r}_j - \mathbf{r}_i\|^2 + \epsilon^2\right)^{3/2}}$$

- **Softening Factor ($\epsilon^2 = 25.0$)**: Prevents numerical overflow and infinite forces when particles pass arbitrarily close ($r \to 0$).
- **Numerical Integration**: Semi-implicit Euler integration advances velocity and position over timestep $\Delta t = 0.01\text{s}$:
  $$\mathbf{v}_i^{(t+1)} = \mathbf{v}_i^{(t)} + \mathbf{a}_i \Delta t$$
  $$\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)} \Delta t$$

### 2. Data Structures
Vector types from CUDA runtime are utilized for memory alignment and coalescing:
- `float4`: Stores position (`x`, `y`), unused `z`, and mass in `w`.
- `float2`: Stores velocity vector (`x`, `y`).

### 3. Shared-Memory Tiling Optimization

In a naive implementation, every thread would read $N$ particle positions from high-latency global memory, generating $\mathcal{O}(N^2)$ global memory loads. 

In [`nbody_cuda.cu`](./nbody_cuda.cu#L32-L76):
- Threads inside a block collaboratively load a chunk of 256 source particles into high-speed on-chip shared memory:
  ```cuda
  __shared__ float4 source_tile[BLOCK_SIZE];
  ```
- All threads in the block reuse these cached positions against their individual target particle.
- This cuts global memory reads by a factor of `BLOCK_SIZE` (256× reduction).

### 4. Barrier Synchronization & Boundary Handling
```cuda
for (int tile = 0; tile < tile_count; ++tile) {
    const int j = tile * blockDim.x + threadIdx.x;
    source_tile[threadIdx.x] = (j < n) ? positions[j] : make_float4(0, 0, 0, 0);
    __syncthreads(); // Wait until tile is loaded

    if (active) {
        // Accumulate forces from tile...
    }
    __syncthreads(); // Wait before next tile overwrites shared memory
}
```
> [!IMPORTANT]
> Both `__syncthreads()` calls sit **outside** the `if (active)` conditional. Because barriers require convergence across all active threads in the block, inactive boundary threads (where $i \ge N$) must still hit every synchronization point to prevent deadlock.

---

## 🚀 Building & Running the CUDA Code

### Prerequisites
- NVIDIA GPU with Compute Capability 3.5+ (Kepler or newer).
- CUDA Toolkit installed (compatible with `nvcc`).
- C++17 compatible host compiler (GCC, Clang, or MSVC).

### 1. Compilation
Compile with optimizations enabled:
```bash
nvcc -O2 -std=c++17 nbody_cuda.cu -o nbody_cuda
```

### 2. Execution
Run the simulation specifying the number of particles $N$ (default is `2048` if omitted):
```bash
# Run with default 2048 particles
./nbody_cuda

# Run with custom particle count (e.g., 10,000 particles)
./nbody_cuda 10000
```

### Sample Output
```text
Completed one timestep for 2048 particles.
Particle 0: x=0.00032 y=0.00032 vx=0.03211 vy=0.03211
```

---

## 🌐 Running the Web Simulation

Open [`nbody_cuda_simulation.html`](./nbody_cuda_simulation.html) directly in any modern browser:

```bash
# macOS
open nbody_cuda_simulation.html

# Linux
xdg-open nbody_cuda_simulation.html

# Or serve via a lightweight HTTP server:
python3 -m http.server 8080
```

### Features of the Web App:
- **Interactive Particle Canvas**: Visualizes softened gravity interactions with glowing orbital trajectories and dynamic boundary reflections.
- **Adjustable Controls**:
  - **Particle Count**: Dynamically scale between 40 and 240 particles.
  - **Speed**: Select between Slow (0.55×), Normal (1.0×), and Fast (1.6×).
  - **Pause / Reset**: Toggle animation or reseed disk orbits.
- **Live Metrics**: Displays pairwise interaction calculations per frame ($N \times (N-1)$) and real-time FPS.
- **Reference Cards**: Explains the CUDA mapping and host lifecycle steps.

---

## 📊 Complexity & Scaling Analysis

| Particle Count ($N$) | Interactions per Step ($N \times (N-1)$) | Work Scale |
|:---------------------|:-----------------------------------------|:-----------|
| **120** (Web Default) | 14,280 | CPU / Canvas realtime |
| **2,048** (CUDA Default) | ~4.19 Million | Sub-millisecond on GPU |
| **10,000** | ~100 Million | Ideal for GPU shared memory |
| **100,000** | ~10 Billion | Saturates modern GPU compute clusters |

While the algorithmic arithmetic remains $\mathcal{O}(N^2)$, the massively parallel execution across thousands of GPU cores combined with shared-memory tiling enables real-time simulation for particle counts that would stall single-threaded CPU implementations.
