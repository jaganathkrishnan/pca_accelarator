// center.cpp
#include "types.h"
#include "params.h"

extern "C" {
void center_data(
    const data_t *X,
    const data_t *mean,
    data_t *Xc
) {
#pragma HLS INTERFACE m_axi port=X    bundle=gmem0
#pragma HLS INTERFACE m_axi port=mean bundle=gmem1
#pragma HLS INTERFACE m_axi port=Xc   bundle=gmem2
#pragma HLS INTERFACE s_axilite port=return

    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D_FEATURES; d++) {
#pragma HLS PIPELINE II=1
            Xc[n * D_FEATURES + d] =
                X[n * D_FEATURES + d] - mean[d];
        }
    }
}
}
