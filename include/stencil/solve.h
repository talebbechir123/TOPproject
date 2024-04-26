#pragma once

#include "mesh.h"


//optimised power function
static inline f64 fpow(f64 base, f64 exp);
void solve_jacobi(mesh_t* A, mesh_t const* B, mesh_t* C);
