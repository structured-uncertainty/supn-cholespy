//
// Created by Neill Campbell on 10/10/2022.
//

#pragma once

#include <iostream>
#include <vector>
#include "cholmod.h"
#include "cuda_driver.h"

#include "cholesky_solver.h"

#include <iostream>
#define vdbg(v) std::cout << #v << " = " << (v) << std::endl;

/**
 * Cholesky Solver Class
 *
 * Takes as in put an arbitrary COO, CSC or CSR matrix, and factorizes it using
 * CHOLMOD. If it receives CUDA arrays as input, it runs an analysis of the
 * factor on the GPU for faster, parallel solving of the triangular system in
 * the solving phase.
 */
template<typename Float>
class SupnCholeskySolver {
public:
  /**
   * Build the solver
   *
   * @param n_rows The number of rows in the matrix
   * @param nnz The number of nonzero entries
   * @param ii Array of row indices if type==COO, column (resp. row) pointer array if type==CSC (resp. CSR)
   * @param ii Array of row indices if type==COO or CSC, column indices if type==CSR
   * @param x Array of nonzero entries
   * @param type The type of the matrix representation. Can be COO, CSC or CSR
   * @param cpu Whether or not to run the CPU version of the solver.
   */
  SupnCholeskySolver(int n_rows, int nnz,
                     int *csr_lower_crows,
                     int *csr_lower_cols,
                     int *csr_lower_indices,
                     int *csc_upper_crows,
                     int *csc_upper_cols,
                     int *csc_upper_indices,
                     int n_raw_values,
                     Float *raw_values,
                     bool cpu);

  ~SupnCholeskySolver();

  // Solve the whole system using the Cholesky factorization on the GPU
  void solve_cuda(CUdeviceptr raw_values, int n_rhs, CUdeviceptr b, CUdeviceptr x, bool skip_lower, bool skip_upper);

  // Solve the whole system using the Cholesky factorization on the CPU
  void solve_cpu(int n_rhs, Float *b, Float *x);

  // Return whether the solver solves on the CPU or on the GPU
  bool is_cpu() { return m_cpu; };

  void debug_print(CUdeviceptr rows, CUdeviceptr cols, CUdeviceptr data_indices, CUdeviceptr raw_data, bool lower) {
//    vdbg(m_cpu);
//    vdbg(m_n);
//    vdbg(m_nnz);
//    vdbg(m_n_rows);
//    vdbg(m_n_entries);
//    vdbg(lower);

    CUdeviceptr rows_d = (lower ? m_lower_rows_d : m_upper_rows_d);
    CUdeviceptr cols_d = (lower ? m_lower_cols_d : m_upper_cols_d);
    CUdeviceptr data_indices_d = (lower ? m_lower_data_indices_d : m_upper_data_indices_d);

    cuda_check(cuMemcpyAsync((CUdeviceptr) rows,
                             (CUdeviceptr) rows_d,
                             (m_n_rows + 1) * sizeof(int), 0));
    cuda_check(cuMemcpyAsync((CUdeviceptr) cols,
                             (CUdeviceptr) cols_d,
                             m_n_entries * sizeof(int), 0));
    cuda_check(cuMemcpyAsync((CUdeviceptr) data_indices,
                             (CUdeviceptr) data_indices_d,
                             m_n_entries * sizeof(int), 0));
    cuda_check(cuMemcpyAsync((CUdeviceptr) raw_data,
                             (CUdeviceptr) m_raw_data_d,
                             m_n_raw_data * sizeof(Float), 0));

  }

  int get_n_entries() {
    return m_n_entries;
  }

  int get_n_rows() {
    return m_n_rows;
  }

  int get_n_raw_data() {
    return m_n_raw_data;
  }

private:

  void perform_cuda_setup(int* permutation_data,
                          int num_rows,
                          int num_nz,
                          int *csr_lower_crows,
                          int *csr_lower_cols,
                          int *csr_lower_data,
                          int *csc_upper_crows,
                          int *csc_upper_cols,
                          int *csc_upper_data);

  // Run the analysis of a triangular matrix obtained through Cholesky
  void analyze_cuda(int n_rows, int n_entries, void *csr_rows, void *csr_cols, int *csr_data_indices, bool lower);

  // Solve one triangular system
  void launch_kernel(bool lower, CUdeviceptr x, CUdeviceptr raw_values_d);

  void launch_kernel_just_upper(CUdeviceptr x, CUdeviceptr raw_values_d, CUdeviceptr b);

  void launch_kernel_just_lower(CUdeviceptr x, CUdeviceptr raw_values_d, CUdeviceptr b);

  int m_nrhs = 0;
  int m_n;
  int m_nnz;

  // CPU or GPU solver?
  bool m_cpu;

  // Pointers used for the analysis, freed if solving on the GPU, kept if solving on the CPU
  cholmod_factor *m_factor;
  cholmod_dense  *m_tmp_chol = nullptr;
  cholmod_common  m_common;

  // Pointers used for the GPU variant

  // Permutation
  CUdeviceptr m_perm_d;

  // CSR Lower triangular
  CUdeviceptr m_lower_rows_d;
  CUdeviceptr m_lower_cols_d;
  CUdeviceptr m_lower_data_indices_d;

  // CSR Upper triangular
  CUdeviceptr m_upper_rows_d;
  CUdeviceptr m_upper_cols_d;
  CUdeviceptr m_upper_data_indices_d;

  // Raw data..
  int m_n_raw_data;
  CUdeviceptr m_raw_data_d;

  // Mask of already processed rows, used for both analysis and solve
  CUdeviceptr m_processed_rows_d;

  // ID of current row being processed by a given block
  CUdeviceptr m_stack_id_d;

  // Sorted indices of rows in each level
  CUdeviceptr m_lower_levels_d;
  CUdeviceptr m_upper_levels_d;

  // Temporary array used for solving the triangular systems in place
  CUdeviceptr m_tmp_d = 0;

  //NDFC
  int m_n_entries = 0;
  int m_n_rows = 0;
};

