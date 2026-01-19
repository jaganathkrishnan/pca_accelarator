// mean.cpp
#include "types.h"
#include "params.h"

extern "C" {
void compute_mean(
    const data_t *X,
    data_t *mean
) {
#pragma HLS INTERFACE m_axi port=X    bundle=gmem0
#pragma HLS INTERFACE m_axi port=mean bundle=gmem1
#pragma HLS INTERFACE s_axilite port=return

    for (int d = 0; d < D_FEATURES; d++) {
        data_t sum = 0;

        for (int n = 0; n < N; n++) {
#pragma HLS PIPELINE II=1
            sum += X[n * D_FEATURES + d];
        }

        mean[d] = sum / (data_t)N;
    }
}
}
