//gemm_tiled.cpp
#include "gemm_tiled.h"

static void load_A(
    const data_t *A,
    data_t A_tile[TM][TK],
    int i0,
    int k0
) {
#pragma HLS INLINE
    for (int i = 0; i < TM; i++) {
        for (int k = 0; k < TK; k++) {
#pragma HLS PIPELINE II=1
            A_tile[i][k] =
                A[(k0 + k) * D_FEATURES + (i0 + i)];
        }
    }
}

static void load_B(
    const data_t *B,
    data_t B_tile[TK][TN],
    int k0,
    int j0
) {
#pragma HLS INLINE
    for (int k = 0; k < TK; k++) {
        for (int j = 0; j < TN; j++) {
#pragma HLS PIPELINE II=1
            B_tile[k][j] =
                B[(k0 + k) * D_FEATURES + (j0 + j)];
        }
    }
}

static void compute(
    data_t A_tile[TM][TK],
    data_t B_tile[TK][TN],
    data_t C_tile[TM][TN]
) {
#pragma HLS INLINE
    for (int i = 0; i < TM; i++) {
        for (int k = 0; k < TK; k++) {
            for (int j = 0; j < TN; j++) {
#pragma HLS UNROLL
                C_tile[i][j] +=
                    A_tile[i][k] * B_tile[k][j];
            }
        }
    }
}

extern "C" {
void gemm_tiled(
    const data_t *A,
    const data_t *B,
    data_t *C
) {
#pragma HLS INTERFACE m_axi port=A bundle=gmem0
#pragma HLS INTERFACE m_axi port=B bundle=gmem1
#pragma HLS INTERFACE m_axi port=C bundle=gmem2
#pragma HLS INTERFACE s_axilite port=return

    data_t A_tile[TM][TK];
    data_t B_tile[TK][TN];
    data_t C_tile[TM][TN];

#pragma HLS ARRAY_PARTITION variable=A_tile dim=2 complete
#pragma HLS ARRAY_PARTITION variable=B_tile dim=1 complete
#pragma HLS ARRAY_PARTITION variable=C_tile dim=2 complete

    for (int i0 = 0; i0 < M; i0 += TM) {
        for (int j0 = 0; j0 < N_GEMM; j0 += TN) {

            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
#pragma HLS PIPELINE II=1
                    C_tile[i][j] = 0;

            for (int k0 = 0; k0 < K; k0 += TK) {
                load_A(A, A_tile, i0, k0);
                load_B(B, B_tile, k0, j0);
                compute(A_tile, B_tile, C_tile);
            }

            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
#pragma HLS PIPELINE II=1
                    C[(i0 + i) * N_GEMM + (j0 + j)] =
                        C_tile[i][j];
        }
    }
}
}
