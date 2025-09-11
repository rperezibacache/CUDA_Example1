// Compile: nvcc -arch=sm_89 -O2 hbridge_sim_fft.cu -lcufft -o hbridge_sim_fft
// Run:     ./hbridge_sim_fft
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>
#include <cufft.h>
#include <cuda_runtime.h>

#define TWO_PI 6.28318530717958647692f

struct HBridgeCellHost {
    float Vdc;
    float duty;
    float phase;
    float f_mod;
};
struct HBridgeCellDevice {
    float Vdc;
    float duty;
    float phase;
    float f_mod;
};

// Triangle carrier in [-1,1]
__device__ __host__ inline float triangle_wave(float t, float fc) {
    float T = 1.0f / fc;
    float x = fmodf(t, T) / T;
    return (x < 0.5f) ? 4.0f * x - 1.0f : -4.0f * x + 3.0f;
}

// Generate SPWM waveform per cell
__global__ void simulate_cells(const HBridgeCellDevice* cells,
                               float* out,
                               int Ncells, int Nsamp,
                               float fs, float fc)
{
    int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= Ncells) return;
    const HBridgeCellDevice c = cells[cellIdx];

    for (int i = 0; i < Nsamp; ++i) {
        float t = (float)i / fs;
        float carrier = triangle_wave(t, fc);
        float ref = c.duty * sinf(TWO_PI * c.f_mod * t + c.phase);
        out[cellIdx * Nsamp + i] = (ref > carrier) ? c.Vdc : -c.Vdc;
    }
}

// Simple helper to print sample snippets
void print_samples(const std::vector<float>& w, int Nsamp, int Ncells) {
    for (int c = 0; c < Ncells; ++c) {
        printf("Cell %d first 12 samples: ", c);
        for (int i = 0; i < 12 && i < Nsamp; ++i)
            printf("%6.1f ", w[c * Nsamp + i]);
        printf("...\n");
    }
}

int main() {
    const int   Ncells = 4;
    const float fs     = 200e3f;   // sample rate
    const int   Nsamp  = 4000;     // FFT-friendly size (power of 2)
    const float fc     = 10e3f;    // carrier

    printf("Simulating %d cells, %d samples, fs %.1f Hz\n",
           Ncells, Nsamp, fs);

    // Host cell definitions
    std::vector<HBridgeCellHost> hcells(Ncells);
    for (int i = 0; i < Ncells; ++i) {
        hcells[i].Vdc   = 400.0f - i * 50.0f;
        hcells[i].duty  = 0.8f  - i * 0.15f;
        hcells[i].phase = 0;//i * (TWO_PI / 8.0f);
        hcells[i].f_mod = 50.0f;
    }

    // Copy cells to device
    std::vector<HBridgeCellDevice> tmp(Ncells);
    for (int i = 0; i < Ncells; ++i) {
        tmp[i] = { hcells[i].Vdc, hcells[i].duty,
                   hcells[i].phase, hcells[i].f_mod };
    }
    HBridgeCellDevice* d_cells;
    cudaMalloc(&d_cells, Ncells * sizeof(HBridgeCellDevice));
    cudaMemcpy(d_cells, tmp.data(), Ncells * sizeof(HBridgeCellDevice),
               cudaMemcpyHostToDevice);

    // Allocate waveform buffer
    float* d_out;
    cudaMalloc(&d_out, Ncells * Nsamp * sizeof(float));
    simulate_cells<<<(Ncells+127)/128, 128>>>(d_cells, d_out,
                                              Ncells, Nsamp, fs, fc);
    cudaDeviceSynchronize();

    // --- cuFFT batched R2C ---
    cufftHandle plan;
    cufftComplex* d_fft;
    cudaMalloc(&d_fft, Ncells * (Nsamp/2 + 1) * sizeof(cufftComplex));
    cufftPlan1d(&plan, Nsamp, CUFFT_R2C, Ncells);
    cufftExecR2C(plan, d_out, d_fft);
    cudaDeviceSynchronize();

    // Copy results back
    std::vector<cufftComplex> h_fft(Ncells * (Nsamp/2 + 1));
    cudaMemcpy(h_fft.data(), d_fft,
               h_fft.size() * sizeof(cufftComplex),
               cudaMemcpyDeviceToHost);

    // Retrieve waveforms too (optional)
    std::vector<float> h_wave(Ncells * Nsamp);
    cudaMemcpy(h_wave.data(), d_out,
               h_wave.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    print_samples(h_wave, Nsamp, Ncells);

    // --- Magnitude spectrum print (fundamental & a few harmonics) ---
    printf("\nFFT results (scaled to V amplitude):\n");
    for (int c = 0; c < Ncells; ++c) {
        printf("Cell %d:\n", c);
        for (int k = 0; k <= 10; ++k) { // first 10 bins
            float freq = k * fs / Nsamp;
            cufftComplex v = h_fft[c*(Nsamp/2 + 1) + k];
            float mag = 2.0f/Nsamp * sqrtf(v.x*v.x + v.y*v.y);
            printf("  %4.1f Hz : %.3f V\n", freq, mag);
        }
    }

    // Save binary waveform (same format as before)
    FILE* fp = fopen("hbridge_output.bin", "wb");
    fwrite(&Ncells, sizeof(int), 1, fp);
    fwrite(&Nsamp, sizeof(int), 1, fp);
    fwrite(&fs,    sizeof(float), 1, fp);
    fwrite(h_wave.data(), sizeof(float), h_wave.size(), fp);
    fclose(fp);
    printf("\nBinary saved: hbridge_output.bin\n");

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_cells);
    cudaFree(d_out);
    cudaFree(d_fft);

    return 0;
}
