//
// Created by Neill Campbell on 10/10/2022.
//

#include "supn_cholesky_solver.h"
#include "cuda_driver.h"
#include <algorithm>
#include <exception>

template <typename Float>
SupnCholeskySolver<Float>::SupnCholeskySolver(int n_rows, int nnz,
                                              int *csr_lower_crows,
                                              int *csr_lower_cols,
                                              int *csr_lower_indices,
                                              int *csc_upper_crows,
                                              int *csc_upper_cols,
                                              int *csc_upper_indices,
                                              int n_raw_values,
                                              Float *raw_values,
                                              bool cpu) : m_n(n_rows), m_nnz(nnz), m_n_raw_data(n_raw_values), m_cpu(cpu) {

    if (m_cpu) {
      throw std::invalid_argument("SupnCholeskySolver: Must operate on the GPU.");
    }

    // Mask of rows already processed
    cuda_check(cuMemAlloc(&m_processed_rows_d, m_n * sizeof(bool)));
    cuda_check(cuMemsetD8Async(m_processed_rows_d, 0, m_n, 0)); // Initialize to all false

    // Row id
    cuda_check(cuMemAlloc(&m_stack_id_d, sizeof(int)));
    cuda_check(cuMemsetD32Async(m_stack_id_d, 0, 1, 0));

    // Copy raw values
    cuda_check(cuMemAlloc(&m_raw_data_d, m_n_raw_data * sizeof(Float)));
    cuda_check(cuMemcpyAsync(m_raw_data_d, raw_values, m_n_raw_data * sizeof(Float), 0));

    // IMPORTANT!!!!
    //
    // HAVE NOT ACCOUNTED FOR PERMUTATIONS - WILL NEED TO FIX IF USED..
    //
    int* permutation_data = (int *)malloc(m_n * sizeof(int));
    for (int i = 0; i < m_n; ++i) {
      permutation_data[i] = i;
    }

    perform_cuda_setup(permutation_data, n_rows, nnz,
                       csr_lower_crows,
                       csr_lower_cols,
                       csr_lower_indices,
                       csc_upper_crows,
                       csc_upper_cols,
                       csc_upper_indices);

    free(permutation_data);
}

template <typename Float>
void SupnCholeskySolver<Float>::perform_cuda_setup(int* permutation_data,
                                                   int num_rows,
                                                   int num_nz,
                                                   int *csr_lower_crows,
                                                   int *csr_lower_cols,
                                                   int *csr_lower_indices,
                                                   int *csc_upper_crows,
                                                   int *csc_upper_cols,
                                                   int *csc_upper_indices) {
  // Copy permutation
  cuda_check(cuMemAlloc(&m_perm_d, m_n * sizeof(int)));
  cuda_check(cuMemcpyAsync(m_perm_d, permutation_data, m_n * sizeof(int), 0));

//  vdbg(m_n);
//  for (int i = 0; i < m_n; ++i) {
//    vdbg(((int *) permutation_data)[i]);
//  }

    int n_entries = num_nz;
    int n_rows = num_rows;

    //NDFC
    m_n_entries = n_entries;
//    vdbg(m_n_entries);
    m_n_rows = n_rows;
//    vdbg(m_n_rows);

//    for (int i = 0; i < n_rows + 1; ++i) {
//      vdbg(((int *) csr_lower_crows)[i]);
//    }
//    for (int i = 0; i < n_entries; ++i) {
//      vdbg(((int *) csr_lower_cols)[i]);
//    }
//    for (int i = 0; i < n_entries; ++i) {
//      vdbg(csr_lower_indices[i]);
//    }

    // actually sending in lower format CSR..
    analyze_cuda(n_rows, n_entries, csr_lower_crows, csr_lower_cols, csr_lower_indices, true);

//    vdbg("analyze_cuda lower finished!");

    analyze_cuda(n_rows, n_entries, csc_upper_crows, csc_upper_cols, csc_upper_indices, false);

//    vdbg("analyze_cuda upper finished!");

}

template <typename Float>
void SupnCholeskySolver<Float>::analyze_cuda(int n_rows, int n_entries, void *csr_rows, void *csr_cols,
                                             int *csr_data_indices, bool lower) {

  CUdeviceptr *rows_d = (lower ? &m_lower_rows_d : &m_upper_rows_d);
  CUdeviceptr *cols_d = (lower ? &m_lower_cols_d : &m_upper_cols_d);
  CUdeviceptr *data_indices_d = (lower ? &m_lower_data_indices_d : &m_upper_data_indices_d);
  CUdeviceptr *levels_d = (lower ? &m_lower_levels_d : &m_upper_levels_d);

//  vdbg(lower);
//  vdbg(((int*)csr_rows)[0]);
//  vdbg(((int*)csr_rows)[1]);
//  vdbg(((int*)csr_rows)[2]);
//  vdbg(((int*)csr_cols)[0]);
//  vdbg(((int*)csr_cols)[1]);
//  vdbg(((int*)csr_cols)[2]);
//  vdbg(csr_data_indices[0]);
//  vdbg(csr_data_indices[1]);
//  vdbg(csr_data_indices[2]);

  // CSR Matrix arrays
  cuda_check(cuMemAlloc(rows_d, (1+n_rows)*sizeof(int)));
  cuda_check(cuMemcpyAsync(*rows_d, csr_rows, (1+n_rows)*sizeof(int), 0));
  cuda_check(cuMemAlloc(cols_d, n_entries*sizeof(int)));
  cuda_check(cuMemcpyAsync(*cols_d, csr_cols, n_entries*sizeof(int), 0));
  cuda_check(cuMemAlloc(data_indices_d, n_entries*sizeof(int)));
  cuda_check(cuMemcpyAsync(*data_indices_d, csr_data_indices, n_entries*sizeof(int), 0));

//  vdbg(data_indices_d);
//  vdbg(*data_indices_d);
//  vdbg(n_rows);
//  vdbg(n_entries);

  // Row i belongs in level level_ind[i]
  CUdeviceptr level_ind_d;
  cuda_check(cuMemAlloc(&level_ind_d, n_rows*sizeof(int)));
  cuda_check(cuMemsetD32Async(level_ind_d, 0, n_rows, 0));

  cuda_check(cuMemsetD8Async(m_processed_rows_d, 0, n_rows, 0)); // Initialize to all false

  CUdeviceptr max_lvl_d;
  cuda_check(cuMemAlloc(&max_lvl_d, sizeof(int)));
  cuda_check(cuMemsetD32Async(max_lvl_d, 0, 1, 0));

  void *args[6] = {
          &n_rows,
          &max_lvl_d,
          &m_processed_rows_d,
          &level_ind_d,
          rows_d,
          cols_d
  };

//  vdbg("about to launch kernel..");

  CUfunction analysis_kernel = (lower ? analysis_lower : analysis_upper);
  cuda_check(cuLaunchKernel(analysis_kernel,
                            n_rows, 1, 1,
                            1, 1, 1,
                            0, 0, args, 0));

//  vdbg("back from launch kernel..");

  int *level_ind_h = (int *) malloc(n_rows*sizeof(int));
  cuda_check(cuMemcpyAsync((CUdeviceptr) level_ind_h, level_ind_d, n_rows*sizeof(int), 0));

//  vdbg(level_ind_h[0]);
//  vdbg(level_ind_h[1]);

  int max_lvl_h = 0;
  cuda_check(cuMemcpyAsync((CUdeviceptr) &max_lvl_h, max_lvl_d, sizeof(int), 0));
  int n_levels = max_lvl_h + 1;

//  vdbg(n_levels);

  // Construct the (sorted) level array
  int *levels_h = (int *) malloc(n_rows*sizeof(int));
  std::vector<int> level_ptr(n_levels + 1, 0);
  // Count the number of rows per level
  for (int i=0; i<n_rows; i++) {
    level_ptr[1+level_ind_h[i]]++;
  }

  // Convert into the list of pointers to the start of each level
  for (int i=0, S=0; i<n_levels; i++){
    S += level_ptr[i+1];
    level_ptr[i+1] = S;
  }

  // Move all rows to their place in the level array
  for (int i=0; i<n_rows; i++) {
    int row_level = level_ind_h[i]; // Row i belongs to level row_level
    levels_h[level_ptr[row_level]] = i;
    level_ptr[row_level]++;
  }

  cuda_check(cuMemAlloc(levels_d, n_rows*sizeof(int)));
  cuda_check(cuMemcpyAsync(*levels_d, levels_h, n_rows*sizeof(int), 0));

//  vdbg("sorted levels..");

  // Free useless stuff
  free(levels_h);
  free(level_ind_h);
  cuda_check(cuMemFree(level_ind_d));

//  vdbg("about to return..");
}

template<typename Float>
void SupnCholeskySolver<Float>::launch_kernel(bool lower, CUdeviceptr x, CUdeviceptr raw_values_d) {
  // Initialize buffers
  cuda_check(cuMemsetD8Async(m_processed_rows_d, 0, m_n, 0)); // Initialize to all false
  cuda_check(cuMemsetD32Async(m_stack_id_d, 0, 1, 0));

  CUdeviceptr rows_d = (lower ? m_lower_rows_d : m_upper_rows_d);
  CUdeviceptr cols_d = (lower ? m_lower_cols_d : m_upper_cols_d);
  CUdeviceptr data_indices_d = (lower ? m_lower_data_indices_d : m_upper_data_indices_d);
  CUdeviceptr levels_d = (lower ? m_lower_levels_d : m_upper_levels_d);

  void *args[12] = {
          &m_nrhs,
          &m_n,
          &m_stack_id_d,
          &levels_d,
          &m_processed_rows_d,
          &rows_d,
          &cols_d,
          &raw_values_d, //&m_raw_data_d,//&data_d,
          &m_tmp_d,
          &x, // This is the array we read from (i.e. b) for lower, and where we write to (i.e. x) for upper
          &m_perm_d,
          &data_indices_d
  };

  CUfunction solve_kernel;
  if(std::is_same_v<Float, float>)
    solve_kernel = (lower ? solve_lower_with_shuffle_float : solve_upper_with_shuffle_float);
  else
    solve_kernel = (lower ? solve_lower_with_shuffle_double : solve_upper_with_shuffle_double);

  cuda_check(cuLaunchKernel(solve_kernel,
                            m_n, 1, 1,
                            128, 1, 1,
                            0, 0, args, 0));
}

template<typename Float>
void SupnCholeskySolver<Float>::launch_kernel_just_upper(CUdeviceptr x, CUdeviceptr raw_values_d, CUdeviceptr b) {
  // Initialize buffers
  cuda_check(cuMemsetD8Async(m_processed_rows_d, 0, m_n, 0)); // Initialize to all false
  cuda_check(cuMemsetD32Async(m_stack_id_d, 0, 1, 0));

  CUdeviceptr rows_d = m_upper_rows_d;
  CUdeviceptr cols_d = m_upper_cols_d;
  CUdeviceptr data_indices_d = m_upper_data_indices_d;
  CUdeviceptr levels_d = m_upper_levels_d;

  // Copy b data to m_tmp_d since it will be overwritten..
  cuda_check(cuMemcpyAsync(m_tmp_d, b, m_nrhs * m_n * sizeof(Float), 0));

  void *args[12] = {
          &m_nrhs,
          &m_n,
          &m_stack_id_d,
          &levels_d,
          &m_processed_rows_d,
          &rows_d,
          &cols_d,
          &raw_values_d, //&m_raw_data_d,//&data_d,
          &m_tmp_d, //b, // Read from for upper..
          &x, // Write to for upper..
          &m_perm_d,
          &data_indices_d
  };

//  vdbg("HAVE NOT ACCOUNTED FOR PERMUTATIONS - WILL NEED TO FIX IF USED..");

  CUfunction solve_kernel;
  if(std::is_same_v<Float, float>)
    solve_kernel = solve_upper_with_shuffle_float;
  else
    solve_kernel = solve_upper_with_shuffle_double;

  cuda_check(cuLaunchKernel(solve_kernel,
                            m_n, 1, 1,
                            128, 1, 1,
                            0, 0, args, 0));
}

template<typename Float>
void SupnCholeskySolver<Float>::launch_kernel_just_lower(CUdeviceptr x, CUdeviceptr raw_values_d, CUdeviceptr b) {
    // Initialize buffers
    cuda_check(cuMemsetD8Async(m_processed_rows_d, 0, m_n, 0)); // Initialize to all false
    cuda_check(cuMemsetD32Async(m_stack_id_d, 0, 1, 0));

    CUdeviceptr rows_d = m_lower_rows_d;
    CUdeviceptr cols_d = m_lower_cols_d;
    CUdeviceptr data_indices_d = m_lower_data_indices_d;
    CUdeviceptr levels_d = m_lower_levels_d;

//    // Copy b data to m_tmp_d since it will be overwritten..
//    cuda_check(cuMemcpyAsync(m_tmp_d, b, m_nrhs * m_n * sizeof(Float), 0));

    void *args[12] = {
            &m_nrhs,
            &m_n,
            &m_stack_id_d,
            &levels_d,
            &m_processed_rows_d,
            &rows_d,
            &cols_d,
            &raw_values_d, //&m_raw_data_d,//&data_d,
            &x, // Write to for lower..
            &b, // Read from for upper..
            &m_perm_d,
            &data_indices_d
    };

//  vdbg("HAVE NOT ACCOUNTED FOR PERMUTATIONS - WILL NEED TO FIX IF USED..");

    CUfunction solve_kernel;
    if(std::is_same_v<Float, float>)
        solve_kernel = solve_lower_with_shuffle_float;
    else
        solve_kernel = solve_lower_with_shuffle_double;

    cuda_check(cuLaunchKernel(solve_kernel,
                              m_n, 1, 1,
                              128, 1, 1,
                              0, 0, args, 0));
}

template <typename Float>
void SupnCholeskySolver<Float>::solve_cuda(CUdeviceptr raw_values, int n_rhs, CUdeviceptr b, CUdeviceptr x,
                                           bool skip_lower, bool skip_upper) {

  if (n_rhs != m_nrhs) {
    if (n_rhs > 128)
      throw std::invalid_argument("The number of RHS should be less than 128.");
    // We need to modify the allocated memory for the solution
    if (m_tmp_d)
      cuda_check(cuMemFree(m_tmp_d));
    cuda_check(cuMemAlloc(&m_tmp_d, n_rhs * m_n * sizeof(Float)));
    m_nrhs = n_rhs;
  }

//  vdbg(m_nrhs);

  CUdeviceptr raw_values_d = raw_values;

  if (skip_lower) {
      launch_kernel_just_upper(x, raw_values_d, b);
  } else if (skip_upper) {
      launch_kernel_just_lower(x, raw_values_d, b);
  } else {
    // Solve lower
    launch_kernel(true, b, raw_values_d);
    // Solve upper
    launch_kernel(false, x, raw_values_d);
  }
}

template<typename Float>
void SupnCholeskySolver<Float>::solve_cpu(int n_rhs, Float *b, Float *x) {

  if (n_rhs != m_nrhs) {
    // We need to modify the allocated memory for the solution
    if (m_tmp_chol)
      cholmod_free_dense(&m_tmp_chol, &m_common);
    m_tmp_chol = cholmod_allocate_dense(m_n,
                                        n_rhs,
                                        m_n,
                                        CHOLMOD_REAL,
                                        &m_common
    );
    m_nrhs = n_rhs;
  }
  // Set cholmod object fields, converting from C style ordering to F style
  double *tmp = (double *)m_tmp_chol->x;
  for (int i=0; i<m_n; ++i)
    for (int j=0; j<n_rhs; ++j)
      tmp[i + j*m_n] = (double) b[i*n_rhs + j];

  cholmod_dense *cholmod_x = cholmod_solve(CHOLMOD_A, m_factor, m_tmp_chol, &m_common);

  double *sol = (double *) cholmod_x->x;
  for (int i=0; i<m_n; ++i)
    for (int j=0; j<n_rhs; ++j)
      x[i*n_rhs + j] = (Float) sol[i + j*m_n];

  cholmod_free_dense(&cholmod_x, &m_common);
}

template <typename Float>
SupnCholeskySolver<Float>::~SupnCholeskySolver() {
  if (m_cpu){
    cholmod_free_factor(&m_factor, &m_common);
    cholmod_finish(&m_common);
  } else {
    scoped_set_context guard(cu_context);

    cuda_check(cuMemFree(m_processed_rows_d));
    cuda_check(cuMemFree(m_stack_id_d));
    cuda_check(cuMemFree(m_perm_d));
    cuda_check(cuMemFree(m_tmp_d));
    cuda_check(cuMemFree(m_lower_rows_d));
    cuda_check(cuMemFree(m_lower_cols_d));
    cuda_check(cuMemFree(m_lower_data_indices_d));
    cuda_check(cuMemFree(m_upper_rows_d));
    cuda_check(cuMemFree(m_upper_cols_d));
    cuda_check(cuMemFree(m_upper_data_indices_d));
    cuda_check(cuMemFree(m_lower_levels_d));
    cuda_check(cuMemFree(m_upper_levels_d));

    cuda_check(cuMemFree(m_raw_data_d));
  }
}

template class SupnCholeskySolver<float>;
template class SupnCholeskySolver<double>;
