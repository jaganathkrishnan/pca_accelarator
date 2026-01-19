//gemm_tiled.h
#ifndef GEMM_TILED_H
#define GEMM_TILED_H

#include "types.h"
#include "params.h"

extern "C" {
void gemm_tiled(
    const data_t *A,
    const data_t *B,
    data_t *C
);
}

#endif
