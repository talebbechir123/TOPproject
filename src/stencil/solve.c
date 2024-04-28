#include "stencil/solve.h"

#include <assert.h>
#include <math.h>


static inline f64 fpow(f64 base, f64 exp) {
   // in c 

    // if exponent is 0, return 1
    if (exp == 0) {
        return 1;
    }

    // if exponent is 1, return base
    if (exp == 1) {
        return base;
    }

    // if exponent is negative, return 1 divided by base raised to the positive exponent
    if (exp < 0) {
        return 1 / fpow(base, -exp);
    }

    // if exponent is even, return base raised to the exponent divided by 2 squared
    if ((int)exp % 2 == 0) {
        return fpow(base, exp / 2) * fpow(base, exp / 2);
    }

    // if exponent is odd, return base times base raised to the exponent minus 1
    return base * fpow(base, exp - 1);
}

void solve_jacobi(mesh_t* A, mesh_t const* B, mesh_t* C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    usz const dim_x = A->dim_x;
    usz const dim_y = A->dim_y;
    usz const dim_z = A->dim_z;
    // for (usz k = STENCIL_ORDER; k < dim_z - STENCIL_ORDER; ++k) {
    //     for (usz j = STENCIL_ORDER; j < dim_y - STENCIL_ORDER; ++j) {
    //         for (usz i = STENCIL_ORDER; i < dim_x - STENCIL_ORDER; ++i) {
    //             C->cells[i][j][k].value = A->cells[i][j][k].value * B->cells[i][j][k].value;

    //             for (usz o = 1; o <= STENCIL_ORDER; ++o) {
    //                 C->cells[i][j][k].value += A->cells[i + o][j][k].value *
    //                                            B->cells[i + o][j][k].value / pow(17.0, (f64)o);
    //                 C->cells[i][j][k].value += A->cells[i - o][j][k].value *
    //                                            B->cells[i - o][j][k].value / pow(17.0, (f64)o);
    //                 C->cells[i][j][k].value += A->cells[i][j + o][k].value *
    //                                            B->cells[i][j + o][k].value / pow(17.0, (f64)o);
    //                 C->cells[i][j][k].value += A->cells[i][j - o][k].value *
    //                                            B->cells[i][j - o][k].value / pow(17.0, (f64)o);
    //                 C->cells[i][j][k].value += A->cells[i][j][k + o].value *
    //                                            B->cells[i][j][k + o].value / pow(17.0, (f64)o);
    //                 C->cells[i][j][k].value += A->cells[i][j][k - o].value *
    //                                            B->cells[i][j][k - o].value / pow(17.0, (f64)o);
    //             }
    //         }
    //     }
    // }
    // Precompute 1/pow(17.0, (f64)o) for each value of o
f64 inverse_pow[STENCIL_ORDER];
for (usz o = 1; o <= STENCIL_ORDER; ++o) {
    inverse_pow[o - 1] = 1.0 / pow(17.0, (f64)o);
}

// Main loop
for (usz k = STENCIL_ORDER; k < dim_z - STENCIL_ORDER; ++k) {
    for (usz j = STENCIL_ORDER; j < dim_y - STENCIL_ORDER; ++j) {
        for (usz i = STENCIL_ORDER; i < dim_x - STENCIL_ORDER; ++i) {
            C->cells[i][j][k].value = A->cells[i][j][k].value * B->cells[i][j][k].value;

            for (usz o = 1; o <= STENCIL_ORDER; ++o) {
                C->cells[i][j][k].value += A->cells[i + o][j][k].value * B->cells[i + o][j][k].value * inverse_pow[o - 1];
                C->cells[i][j][k].value += A->cells[i - o][j][k].value * B->cells[i - o][j][k].value * inverse_pow[o - 1];
                C->cells[i][j][k].value += A->cells[i][j + o][k].value * B->cells[i][j + o][k].value * inverse_pow[o - 1];
                C->cells[i][j][k].value += A->cells[i][j - o][k].value * B->cells[i][j - o][k].value * inverse_pow[o - 1];
                C->cells[i][j][k].value += A->cells[i][j][k + o].value * B->cells[i][j][k + o].value * inverse_pow[o - 1];
                C->cells[i][j][k].value += A->cells[i][j][k - o].value * B->cells[i][j][k - o].value * inverse_pow[o - 1];
            }
        }
    }
}

    mesh_copy_core(A, C);
}





void solve_jacobi_CB(mesh_t* A, mesh_t const* B, mesh_t* C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    usz const dim_x = A->dim_x;
    usz const dim_y = A->dim_y;
    usz const dim_z = A->dim_z;

    // Cache line blocking parameters
    const int block_size_x = 8; // Number of cells in x-direction per block
    const int block_size_y = 8; // Number of cells in y-direction per block
    const int block_size_z = 8; // Number of cells in z-direction per block

    // Precompute 1/pow(17.0, (f64)o) for each value of o
    f64 inverse_pow[STENCIL_ORDER];
    for (usz o = 1; o <= STENCIL_ORDER; ++o) {
        inverse_pow[o - 1] = 1.0 / pow(17.0, (f64)o);
    }

    // Main loop with cache line blocking
    for (usz bk = STENCIL_ORDER; bk < dim_z - STENCIL_ORDER; bk += block_size_z) {
        for (usz bj = STENCIL_ORDER; bj < dim_y - STENCIL_ORDER; bj += block_size_y) {
            for (usz bi = STENCIL_ORDER; bi < dim_x - STENCIL_ORDER; bi += block_size_x) {
                // Process each block
                for (usz k = bk; k < min(dim_z - STENCIL_ORDER, bk + block_size_z); ++k) {
                    for (usz j = bj; j < min(dim_y - STENCIL_ORDER, bj + block_size_y); ++j) {
                        for (usz i = bi; i < min(dim_x - STENCIL_ORDER, bi + block_size_x); ++i) {
                            C->cells[i][j][k].value = A->cells[i][j][k].value * B->cells[i][j][k].value;

                            for (usz o = 1; o <= STENCIL_ORDER; ++o) {
                                C->cells[i][j][k].value += A->cells[i + o][j][k].value * B->cells[i + o][j][k].value * inverse_pow[o - 1];
                                C->cells[i][j][k].value += A->cells[i - o][j][k].value * B->cells[i - o][j][k].value * inverse_pow[o - 1];
                                C->cells[i][j][k].value += A->cells[i][j + o][k].value * B->cells[i][j + o][k].value * inverse_pow[o - 1];
                                C->cells[i][j][k].value += A->cells[i][j - o][k].value * B->cells[i][j - o][k].value * inverse_pow[o - 1];
                                C->cells[i][j][k].value += A->cells[i][j][k + o].value * B->cells[i][j][k + o].value * inverse_pow[o - 1];
                                C->cells[i][j][k].value += A->cells[i][j][k - o].value * B->cells[i][j][k - o].value * inverse_pow[o - 1];
                            }
                        }
                    }
                }
            }
        }
    }

    mesh_copy_core(A, C);
}
