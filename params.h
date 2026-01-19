//params.h
#ifndef PARAMS_H
#define PARAMS_H

// Data dimensions
#define N 4              // number of samples
#define D_FEATURES 3      // number of features

// GEMM dimensions
#define M D_FEATURES       // rows of A^T
#define K N                // columns of A^T / rows of B
#define N_GEMM D_FEATURES  // columns of B

// Tile sizes (safe defaults)
#define TM 1
#define TN 1
#define TK 1

#endif
