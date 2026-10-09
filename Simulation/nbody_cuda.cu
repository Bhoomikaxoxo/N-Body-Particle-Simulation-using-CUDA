// Educational 2D N-body CUDA example: one simulation timestep.
// Build with: nvcc -O2 -std=c++17 nbody_cuda.cu -o nbody_cuda
// Run with:   ./nbody_cuda [particle_count]
//
// Every target particle i accumulates gravity from all source particles j.
// The shared-memory tile reuses source positions within a block. This remains
// O(N^2) arithmetic; shared memory reduces repeated global-memory loads.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int BLOCK_SIZE = 256;
constexpr float G = 1.0f;
constexpr float EPS2 = 25.0f;  // softening avoids singular force at r = 0
constexpr float DT = 0.01f;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t error = (call);                                               \
    if (error != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error));                               \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

// float4 stores x, y, unused z, and mass in w.
__global__ void nbody_step(const float4* positions, const float2* velocities,
                           float4* next_positions, float2* next_velocities,
                           int n, float dt) {
  __shared__ float4 source_tile[BLOCK_SIZE];

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = i < n;
  float ax = 0.0f;
  float ay = 0.0f;
  float4 target;
  if (active) target = positions[i];

  const int tile_count = (n + blockDim.x - 1) / blockDim.x;
  for (int tile = 0; tile < tile_count; ++tile) {
    const int j = tile * blockDim.x + threadIdx.x;
    source_tile[threadIdx.x] = (j < n) ? positions[j] : make_float4(0, 0, 0, 0);
    __syncthreads();

    if (active) {
      const int valid = min(blockDim.x, n - tile * blockDim.x);
      for (int k = 0; k < valid; ++k) {
        const float4 source = source_tile[k];
        const float dx = source.x - target.x;
        const float dy = source.y - target.y;
        const float r2 = dx * dx + dy * dy + EPS2;
        const float inv_r = rsqrtf(r2);
        const float scale = G * source.w * inv_r * inv_r * inv_r;
        ax += scale * dx;
        ay += scale * dy;
      }
    }
    // All threads, including inactive boundary threads, must reach both barriers.
    __syncthreads();
  }

  if (active) {
    float2 velocity = velocities[i];
    velocity.x += ax * dt;
    velocity.y += ay * dt;
    target.x += velocity.x * dt;
    target.y += velocity.y * dt;
    next_velocities[i] = velocity;
    next_positions[i] = target;
  }
}

int main(int argc, char** argv) {
  const int n = (argc > 1) ? std::atoi(argv[1]) : 2048;
  if (n <= 0) {
    std::fprintf(stderr, "particle_count must be positive\n");
    return EXIT_FAILURE;
  }

  const size_t position_bytes = static_cast<size_t>(n) * sizeof(float4);
  const size_t velocity_bytes = static_cast<size_t>(n) * sizeof(float2);

  // Step 1: allocate CPU host arrays and GPU device arrays.
  auto* h_positions = static_cast<float4*>(std::malloc(position_bytes));
  auto* h_velocities = static_cast<float2*>(std::malloc(velocity_bytes));
  auto* d_positions = static_cast<float4*>(nullptr);
  auto* d_velocities = static_cast<float2*>(nullptr);
  auto* d_next_positions = static_cast<float4*>(nullptr);
  auto* d_next_velocities = static_cast<float2*>(nullptr);
  if (!h_positions || !h_velocities) {
    std::fprintf(stderr, "host allocation failed for N=%d\n", n);
    std::free(h_positions);
    std::free(h_velocities);
    return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaMalloc(&d_positions, position_bytes));
  CUDA_CHECK(cudaMalloc(&d_velocities, velocity_bytes));
  CUDA_CHECK(cudaMalloc(&d_next_positions, position_bytes));
  CUDA_CHECK(cudaMalloc(&d_next_velocities, velocity_bytes));

  // Initial particles on a grid with zero initial velocity.
  const int side = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
  for (int i = 0; i < n; ++i) {
    const int row = i / side;
    const int col = i % side;
    h_positions[i] = make_float4(static_cast<float>(col) * 2.0f,
                                 static_cast<float>(row) * 2.0f, 0.0f, 1.0f);
    h_velocities[i] = make_float2(0.0f, 0.0f);
  }

  // Step 2: copy starting state from CPU host memory to GPU device memory.
  CUDA_CHECK(cudaMemcpy(d_positions, h_positions, position_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_velocities, h_velocities, velocity_bytes,
                        cudaMemcpyHostToDevice));

  // Step 3: launch enough threads to cover N; the kernel guards index >= N.
  const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  nbody_step<<<blocks, BLOCK_SIZE>>>(d_positions, d_velocities,
                                     d_next_positions, d_next_velocities, n, DT);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Step 4: copy the updated state from GPU device memory back to CPU memory.
  CUDA_CHECK(cudaMemcpy(h_positions, d_next_positions, position_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_velocities, d_next_velocities, velocity_bytes,
                        cudaMemcpyDeviceToHost));

  std::printf("Completed one timestep for %d particles.\n", n);
  std::printf("Particle 0: x=%.5f y=%.5f vx=%.5f vy=%.5f\n",
              h_positions[0].x, h_positions[0].y,
              h_velocities[0].x, h_velocities[0].y);

  // Step 5: release every host and device allocation.
  std::free(h_positions);
  std::free(h_velocities);
  CUDA_CHECK(cudaFree(d_positions));
  CUDA_CHECK(cudaFree(d_velocities));
  CUDA_CHECK(cudaFree(d_next_positions));
  CUDA_CHECK(cudaFree(d_next_velocities));
  return EXIT_SUCCESS;
}
