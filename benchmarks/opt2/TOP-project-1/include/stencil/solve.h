#pragma once

#include "mesh.h"

#define min(a, b) ((a) < (b) ? (a) : (b))
void solve_jacobi_original(mesh_t* A, mesh_t const* B, mesh_t* C);
void solve_jacobi(mesh_t* A, mesh_t const* B, mesh_t* C);
void solve_jacobi_CB(mesh_t* A, mesh_t const* B, mesh_t* C);