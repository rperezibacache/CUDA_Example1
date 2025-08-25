#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Second order system: m*x'' + c*x' + k*x = u
// State-space representation: x[k+1] = A*x[k] + B*u[k]
// where x = [position; velocity]

__global__ void simulateSystem(float* A, float* B, float* x0, float* u, float* x_history, int num_steps, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 10000) return;  // We'll run 10,000 simulations
    
    // System parameters for each simulation (could be different)
    float m = 1.0f + idx * 0.0001f;  // Mass varies slightly
    float c = 0.1f + idx * 0.00001f; // Damping varies slightly  
    float k = 1.0f + idx * 0.0001f;  // Stiffness varies slightly
    
    // State-space matrices (discrete time)
    float A_local[4], B_local[2];
    
    // Continuous-time matrices
    // A_cont = [0, 1; -k/m, -c/m]
    // B_cont = [0; 1/m]
    
    // Discrete-time approximation (Euler method)
    A_local[0] = 1.0f;
    A_local[1] = dt;
    A_local[2] = -k/m * dt;
    A_local[3] = 1.0f - c/m * dt;
    
    B_local[0] = 0.0f;
    B_local[1] = dt / m;
    
    // Initial state
    float x[2] = {x0[2*idx], x0[2*idx + 1]};
    
    // Store initial state
    x_history[idx * (num_steps + 1) * 2 + 0] = x[0];  // position
    x_history[idx * (num_steps + 1) * 2 + 1] = x[1];  // velocity
    
    // Simulate
    for (int step = 0; step < num_steps; step++) {
        float u_k = u[step];  // Input at this timestep
        
        // x[k+1] = A*x[k] + B*u[k]
        float x_new[2];
        x_new[0] = A_local[0] * x[0] + A_local[1] * x[1] + B_local[0] * u_k;
        x_new[1] = A_local[2] * x[0] + A_local[3] * x[1] + B_local[1] * u_k;
        
        x[0] = x_new[0];
        x[1] = x_new[1];
        
        // Store state
        x_history[idx * (num_steps + 1) * 2 + (step + 1) * 2 + 0] = x[0];
        x_history[idx * (num_steps + 1) * 2 + (step + 1) * 2 + 1] = x[1];
    }
}

int main() {
    const int num_simulations = 10000;
    const int num_steps = 100000;
    const float dt = 0.001f;
    
    size_t state_size = num_simulations * 2 * sizeof(float);
    size_t history_size = num_simulations * (num_steps + 1) * 2 * sizeof(float);
    size_t input_size = num_steps * sizeof(float);
    
    // Host arrays
    float *h_x0 = (float*)malloc(state_size);
    float *h_u = (float*)malloc(input_size);
    float *h_x_history = (float*)malloc(history_size);
    
    // Initialize initial conditions and input
    for (int i = 0; i < num_simulations; i++) {
        h_x0[2*i] = 0.1f * (i % 10);      // Initial position
        h_x0[2*i + 1] = 0.0f;             // Initial velocity
    }
    
    for (int i = 0; i < num_steps; i++) {
        // Step input: 1 for first half, 0 for second half
        h_u[i] = (i < num_steps / 2) ? 0.0f : 0.0f;
    }
    
    // Device arrays
    float *d_x0, *d_u, *d_x_history;
    cudaMalloc(&d_x0, state_size);
    cudaMalloc(&d_u, input_size);
    cudaMalloc(&d_x_history, history_size);
    
    // Copy to device
    cudaMemcpy(d_x0, h_x0, state_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u, input_size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_simulations + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Running %d simulations with %d steps each...\n", num_simulations, num_steps);
    simulateSystem<<<blocksPerGrid, threadsPerBlock>>>(NULL, NULL, d_x0, d_u, d_x_history, num_steps, dt);
    
    // Copy results back to host
    cudaMemcpy(h_x_history, d_x_history, history_size, cudaMemcpyDeviceToHost);
    
    // Save results to binary file for Python
    FILE *file = fopen("simulation_results.bin", "wb");
    if (file) {
        // Write metadata
        int metadata[3] = {num_simulations, num_steps, 2};
        fwrite(metadata, sizeof(int), 3, file);
        
        // Write time array
        float *time = (float*)malloc(num_steps * sizeof(float));
        for (int i = 0; i <= num_steps; i++) time[i] = i * dt;
        fwrite(time, sizeof(float), num_steps + 1, file);
        free(time);
        
        // Write input signal
        fwrite(h_u, sizeof(float), num_steps, file);
        
        // Write results
        fwrite(h_x_history, sizeof(float), num_simulations * (num_steps + 1) * 2, file);
        fclose(file);
        printf("Results saved to simulation_results.bin\n");
    }
    
    // Cleanup
    free(h_x0);
    free(h_u);
    free(h_x_history);
    cudaFree(d_x0);
    cudaFree(d_u);
    cudaFree(d_x_history);
    
    return 0;
}