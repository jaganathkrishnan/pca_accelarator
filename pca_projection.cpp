//pca_projection.cpp
#include "types.h"
#include "params.h"

extern "C" {
void project_pca(
    const data_t *Xc,      // N x D
    const data_t *eigvec,  // D
    data_t *Y              // N
) {
#pragma HLS INTERFACE m_axi     port=Xc     bundle=gmem0
#pragma HLS INTERFACE m_axi     port=eigvec bundle=gmem1
#pragma HLS INTERFACE m_axi     port=Y      bundle=gmem2

#pragma HLS INTERFACE s_axilite port=Xc
#pragma HLS INTERFACE s_axilite port=eigvec
#pragma HLS INTERFACE s_axilite port=Y
#pragma HLS INTERFACE s_axilite port=return

    for (int n = 0; n < N; n++) {
#pragma HLS PIPELINE II=1
        data_t acc = 0;
        for (int d = 0; d < D_FEATURES; d++) {
#pragma HLS UNROLL
            acc += Xc[n * D_FEATURES + d] * eigvec[d];
        }
        Y[n] = acc;
    }
}
}
