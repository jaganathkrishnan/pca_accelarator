//pca_power_iteration.cpp
#include <cmath>
#include "types.h"
#include "params.h"

// Normalize a vector
static void normalize(data_t *v) {
    data_t norm = 0;
    for (int i = 0; i < D_FEATURES; i++)
        norm += v[i] * v[i];

    norm = std::sqrt(norm);

    for (int i = 0; i < D_FEATURES; i++)
        v[i] /= norm;
}

// Power Iteration to find top eigenvector
void power_iteration(
    const data_t *Cov,   // D x D
    data_t *eigvec,      // D
    int iters = 20
) {
    data_t temp[D_FEATURES];

    // Initial guess (can be anything non-zero)
    for (int i = 0; i < D_FEATURES; i++)
        eigvec[i] = 1.0f;

    normalize(eigvec);

    for (int iter = 0; iter < iters; iter++) {

        // temp = Cov * eigvec
        for (int i = 0; i < D_FEATURES; i++) {
            temp[i] = 0;
            for (int j = 0; j < D_FEATURES; j++) {
                temp[i] += Cov[i * D_FEATURES + j] * eigvec[j];
            }
        }

        // eigvec = temp / ||temp||
        for (int i = 0; i < D_FEATURES; i++)
            eigvec[i] = temp[i];

        normalize(eigvec);
    }
}
