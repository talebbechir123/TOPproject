#include "stencil/solve.h"
#include <immintrin.h>
#include <assert.h>
#include <math.h>


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
                C->cells[i][j][k].value += (A->cells[i + o][j][k].value * B->cells[i + o][j][k].value
                + A->cells[i - o][j][k].value * B->cells[i - o][j][k].value 
                + A->cells[i][j + o][k].value * B->cells[i][j + o][k].value 
                + A->cells[i][j - o][k].value * B->cells[i][j - o][k].value 
                + A->cells[i][j][k + o].value * B->cells[i][j][k + o].value 
                + A->cells[i][j][k - o].value * B->cells[i][j][k - o].value)*inverse_pow[o - 1];
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
                                C->cells[i][j][k].value += (A->cells[i + o][j][k].value * B->cells[i + o][j][k].value
                                + A->cells[i - o][j][k].value * B->cells[i - o][j][k].value 
                                + A->cells[i][j + o][k].value * B->cells[i][j + o][k].value 
                                + A->cells[i][j - o][k].value * B->cells[i][j - o][k].value 
                                + A->cells[i][j][k + o].value * B->cells[i][j][k + o].value 
                                + A->cells[i][j][k - o].value * B->cells[i][j][k - o].value)*inverse_pow[o - 1];
                            }
                        }
                    }
                }
            }
        }
    }

    mesh_copy_core(A, C);
}


$


void solve_jacobi_vectorized(mesh_t* A, mesh_t const* B, mesh_t* C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    // Precompute 1/pow(17.0, (f64)o) for each value of o
    double inverse_pow[STENCIL_ORDER];
    for (size_t o = 1; o <= STENCIL_ORDER; ++o) {
        inverse_pow[o - 1] = 1.0 / pow(17.0, (double)o);
    }

    // Main loop
    for (size_t k = STENCIL_ORDER; k < A->dim_z - STENCIL_ORDER; ++k) {
        for (size_t j = STENCIL_ORDER; j < A->dim_y - STENCIL_ORDER; ++j) {
            for (size_t i = STENCIL_ORDER; i < A->dim_x - STENCIL_ORDER; ++i) {
                // Initialize SIMD vectors
                __m256d acc = _mm256_setzero_pd();
                __m256d pow_vals = _mm256_loadu_pd(&inverse_pow[0]);

                // Compute the contribution from core cell
                acc = _mm256_mul_pd(acc, _mm256_loadu_pd(&A->cells[i][j][k].value));
                acc = _mm256_mul_pd(acc, _mm256_loadu_pd(&B->cells[i][j][k].value));

                // Compute the contribution from neighboring cells using SIMD
                for (size_t o = 1; o <= STENCIL_ORDER; ++o) {
                    // X-axis offset
                    __m256d a_vals_x = _mm256_loadu_pd(&A->cells[i + o][j][k].value);
                    __m256d b_vals_x = _mm256_loadu_pd(&B->cells[i + o][j][k].value);
                    __m256d inverse_pow_vals_x = _mm256_broadcast_sd(&inverse_pow[o - 1]);
                    acc = _mm256_fmadd_pd(a_vals_x, b_vals_x, _mm256_mul_pd(acc, inverse_pow_vals_x));

                    // Y-axis offset
                    __m256d a_vals_y = _mm256_loadu_pd(&A->cells[i][j + o][k].value);
                    __m256d b_vals_y = _mm256_loadu_pd(&B->cells[i][j + o][k].value);
                    __m256d inverse_pow_vals_y = _mm256_broadcast_sd(&inverse_pow[o - 1]);
                    acc = _mm256_fmadd_pd(a_vals_y, b_vals_y, _mm256_mul_pd(acc, inverse_pow_vals_y));

                    // Z-axis offset
                    __m256d a_vals_z = _mm256_loadu_pd(&A->cells[i][j][k + o].value);
                    __m256d b_vals_z = _mm256_loadu_pd(&B->cells[i][j][k + o].value);
                    __m256d inverse_pow_vals_z = _mm256_broadcast_sd(&inverse_pow[o - 1]);
                    acc = _mm256_fmadd_pd(a_vals_z, b_vals_z, _mm256_mul_pd(acc, inverse_pow_vals_z));
                }

                // Store the result back to memory
                _mm256_storeu_pd(&C->cells[i][j][k].value, acc);
            }
        }
    }
}
