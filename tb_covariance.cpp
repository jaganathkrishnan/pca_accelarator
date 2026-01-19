#include <iostream>
#include <iomanip>
#include <cmath>
#include "types.h"
#include "params.h"

extern "C" {
void compute_mean(const data_t *X, data_t *mean);
void center_data(const data_t *X, const data_t *mean, data_t *Xc);
void gemm_tiled(const data_t *A, const data_t *B, data_t *C);
}

int main() {

    data_t X[N * D_FEATURES];
    data_t mean[D_FEATURES];
    data_t Xc[N * D_FEATURES];
    data_t Cov[D_FEATURES * D_FEATURES];

    // -------------------------
    // Input matrix (4x3)
    // -------------------------
    data_t X_init[N][D_FEATURES] = {
        { 1,  2,  3},
        { 4,  5,  6},
        { 7,  8,  9},
        {10, 11, 12}
    };

    for (int n = 0; n < N; n++)
        for (int d = 0; d < D_FEATURES; d++)
            X[n * D_FEATURES + d] = X_init[n][d];

    // -------------------------
    // Run kernels
    // -------------------------
    compute_mean(X, mean);
    center_data(X, mean, Xc);
    gemm_tiled(Xc, Xc, Cov);

    // -------------------------
    // Print input
    // -------------------------
    std::cout << "\nInput X:\n";
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D_FEATURES; d++)
            std::cout << std::setw(6) << X[n * D_FEATURES + d];
        std::cout << "\n";
    }

    // -------------------------
    // Print mean
    // -------------------------
    std::cout << "\nMean:\n";
    for (int d = 0; d < D_FEATURES; d++)
        std::cout << std::setw(6) << mean[d];
    std::cout << "\n";

    // -------------------------
    // Print centered data
    // -------------------------
    std::cout << "\nCentered Xc:\n";
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D_FEATURES; d++)
            std::cout << std::setw(6) << Xc[n * D_FEATURES + d];
        std::cout << "\n";
    }

    // -------------------------
    // Print covariance
    // -------------------------
    std::cout << "\nCovariance = Xc^T * Xc:\n";
    for (int i = 0; i < D_FEATURES; i++) {
        for (int j = 0; j < D_FEATURES; j++)
            std::cout << std::setw(8) << Cov[i * D_FEATURES + j];
        std::cout << "\n";
    }

    return 0;
}
