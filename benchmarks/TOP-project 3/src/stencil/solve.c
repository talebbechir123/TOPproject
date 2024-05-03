#include "stencil/solve.h"

#include <assert.h>
#include <math.h>


double dpow(double base, int exp) {
    double result = 1.0;
    while (exp != 0) {
        if (exp & 1) {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }
    return result;
}




void solve_jacobi_original(mesh_t* A, mesh_t const* B, mesh_t* C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    usz const dim_x = A->dim_x;
    usz const dim_y = A->dim_y;
    usz const dim_z = A->dim_z;


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
                + A->cells[i][j][k - o].value * B->cells[i][j][k - o].value)/pow(17.0, (f64)o);
            }
        }
    }
}

    mesh_copy_core(A, C);
}







void solve_jacobi(mesh_t* A, mesh_t const* B, mesh_t* C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    usz const dim_x = A->dim_x;
    usz const dim_y = A->dim_y;
    usz const dim_z = A->dim_z;

    // Precompute 1/pow(17.0, (f64)o) for each value of o
f64 inverse_pow[STENCIL_ORDER];
for (usz o = 1; o <= STENCIL_ORDER; ++o) {
    inverse_pow[o - 1] = 1.0 / dpow(17.0, (f64)o);
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
    const int block_size_x = 64; // Number of cells in x-direction per block
    const int block_size_y = 64; // Number of cells in y-direction per block
    const int block_size_z = 64; // Number of cells in z-direction per block

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





