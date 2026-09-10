#include "cholesky_solver.h"
#include "supn_cholesky_solver.h"
#include "docstr.h"
#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)

void cuda_check_impl(CUresult errval, const char *file, const int line);
namespace nb = nanobind;

// void cuda_check_impl(CUresult errval, const char *file, const int line);

template <typename Float>
void declare_cholesky(nb::module_ &m, const std::string &typestr, const char *docstr) {
    using Class = CholeskySolver<Float>;
    std::string class_name = std::string("CholeskySolver") + typestr;
    nb::class_<Class>(m, class_name.c_str(), docstr)
        .def("__init__", [](Class *self,
                            uint32_t n_rows,
                            nb::tensor<int32_t, nb::shape<nb::any>, nb::c_contig> ii,
                            nb::tensor<int32_t, nb::shape<nb::any>, nb::c_contig> jj,
                            nb::tensor<double, nb::shape<nb::any>, nb::c_contig> x,
                            MatrixType type) {

            if (type == MatrixType::COO){
                if (ii.shape(0) != jj.shape(0))
                    throw std::invalid_argument("Sparse COO matrix: the two index arrays should have the same size.");
                if (ii.shape(0) != x.shape(0))
                    throw std::invalid_argument("Sparse COO matrix: the index and data arrays should have the same size.");
            } else if (type == MatrixType::CSR) {
                if (jj.shape(0) != x.shape(0))
                    throw std::invalid_argument("Sparse CSR matrix: the column index and data arrays should have the same size.");
                if (ii.shape(0) != n_rows+1)
                    throw std::invalid_argument("Sparse CSR matrix: Invalid size for row pointer array.");
            } else {
                if (jj.shape(0) != x.shape(0))
                    throw std::invalid_argument("Sparse CSC matrix: the row index and data arrays should have the same size.");
                if (ii.shape(0) != n_rows+1)
                    throw std::invalid_argument("Sparse CSC matrix: Invalid size for column pointer array.");
            }
            if (ii.device_type() != jj.device_type() || ii.device_type() != x.device_type())
                throw std::invalid_argument("All input tensors should be on the same device!");

            if (ii.device_type() == nb::device::cuda::value) {
                // GPU init

                // Initialize CUDA and load the kernels if not already done
                init_cuda();

                scoped_set_context guard(cu_context);

                int *indices_a = (int *) malloc(ii.shape(0)*sizeof(int));
                int *indices_b = (int *) malloc(jj.shape(0)*sizeof(int));
                double *data = (double *) malloc(x.shape(0)*sizeof(double));

                cuda_check(cuMemcpyAsync((CUdeviceptr) indices_a, (CUdeviceptr) ii.data(), ii.shape(0)*sizeof(int), 0));
                cuda_check(cuMemcpyAsync((CUdeviceptr) indices_b, (CUdeviceptr) jj.data(), jj.shape(0)*sizeof(int), 0));
                cuda_check(cuMemcpyAsync((CUdeviceptr) data, (CUdeviceptr) x.data(), x.shape(0)*sizeof(double), 0));

                new (self) Class(n_rows, x.shape(0), indices_a, indices_b, data, type, false);

                free(indices_a);
                free(indices_b);
                free(data);
            } else if (ii.device_type() == nb::device::cpu::value) {
                // CPU init
                new (self) Class(n_rows, x.shape(0), (int *) ii.data(), (int *) jj.data(), (double *) x.data(), type, true);
            } else
                throw std::invalid_argument("Unsupported input device! Only CPU and CUDA arrays are supported.");
        },
        nb::arg("n_rows"),
        nb::arg("ii"),
        nb::arg("jj"),
        nb::arg("x"),
        nb::arg("type"),
        doc_constructor)
        .def("solve", [](Class &self,
                        nb::tensor<Float, nb::c_contig> b,
                        nb::tensor<Float, nb::c_contig> x){
            if (b.ndim() != 1 && b.ndim() != 2)
                throw std::invalid_argument("Expected 1D or 2D tensors as input.");
            if (b.shape(0) != x.shape(0) || (b.ndim() == 2 && b.shape(1) != x.shape(1)))
                throw std::invalid_argument("x and b should have the same dimensions.");
            if (b.device_type() != x.device_type())
                throw std::invalid_argument("x and b should be on the same device.");

            // CPU solve
            if (b.device_type() == nb::device::cpu::value) {
                if (!self.is_cpu())
                    throw std::invalid_argument("Input device is CPU but the solver was initialized for CUDA.");

                self.solve_cpu(b.ndim()==2 ? b.shape(1) : 1, (Float *) b.data(), (Float *) x.data());
            }
            // CUDA solve
            else if (b.device_type() == nb::device::cuda::value) {
                if (self.is_cpu())
                    throw std::invalid_argument("Input device is CUDA but the solver was initialized for CPU.");

                scoped_set_context guard(cu_context);
                self.solve_cuda(b.ndim()==2 ? b.shape(1) : 1, (CUdeviceptr) b.data(), (CUdeviceptr) x.data());
            }
            else
                throw std::invalid_argument("Unsupported input device! Only CPU and CUDA arrays are supported.");
        },
        nb::arg("b").noconvert(),
        nb::arg("x").noconvert(),
        doc_solve);
}


template <typename Float>
void declare_supn_cholesky(nb::module_ &m, const std::string &typestr, const char *docstr) {
  using Class = SupnCholeskySolver<Float>;
  std::string class_name = std::string("SupnCholeskySolver") + typestr;
  nb::class_<Class>(m, class_name.c_str(), docstr)
          .def("__init__", [](Class *self,
                              uint32_t n_rows,
                              nb::tensor<int32_t, nb::shape<nb::any>, nb::c_contig> csr_lower_crows,
                              nb::tensor<int32_t, nb::shape<nb::any>, nb::c_contig> csr_lower_cols,
                              nb::tensor<int32_t, nb::shape<nb::any>, nb::c_contig> csr_lower_indices,
                              nb::tensor<int32_t, nb::shape<nb::any>, nb::c_contig> csc_upper_crows,
                              nb::tensor<int32_t, nb::shape<nb::any>, nb::c_contig> csc_upper_cols,
                              nb::tensor<int32_t, nb::shape<nb::any>, nb::c_contig> csc_upper_indices,
                              nb::tensor<Float, nb::shape<nb::any>, nb::c_contig> raw_values) {

                   if (csr_lower_cols.shape(0) != csr_lower_indices.shape(0))
                     throw std::invalid_argument("Sparse CSC matrix: the row index and data arrays should have the same size.");
                   if (csr_lower_crows.shape(0) != n_rows+1)
                     throw std::invalid_argument("Sparse CSC matrix: Invalid size for column pointer array.");

                 if (csr_lower_crows.device_type() != csr_lower_cols.device_type() ||
                     csr_lower_crows.device_type() != csr_lower_indices.device_type())
                   throw std::invalid_argument("All input tensors should be on the same device!");

                 if (csc_upper_cols.shape(0) != csc_upper_indices.shape(0))
                   throw std::invalid_argument("Sparse CSC matrix: the row index and data arrays should have the same size.");
                 if (csc_upper_crows.shape(0) != n_rows+1)
                   throw std::invalid_argument("Sparse CSC matrix: Invalid size for column pointer array.");

                 if (csc_upper_crows.device_type() != csc_upper_cols.device_type() ||
                         csc_upper_crows.device_type() != csc_upper_indices.device_type())
                   throw std::invalid_argument("All input tensors should be on the same device!");

                 if (csr_lower_crows.device_type() == nb::device::cuda::value) {

                   // GPU init

                   // Initialize CUDA and load the kernels if not already done
                   init_cuda();

                   scoped_set_context guard(cu_context);

                   int *lower_indices_a = (int *) malloc(csr_lower_crows.shape(0)*sizeof(int));
                   int *lower_indices_b = (int *) malloc(csr_lower_cols.shape(0)*sizeof(int));
                   int *lower_value_indices = (int *) malloc(csr_lower_indices.shape(0)*sizeof(int));

                   int *upper_indices_a = (int *) malloc(csc_upper_crows.shape(0)*sizeof(int));
                   int *upper_indices_b = (int *) malloc(csc_upper_cols.shape(0)*sizeof(int));
                   int *upper_value_indices = (int *) malloc(csc_upper_indices.shape(0)*sizeof(int));

                   cuda_check(cuMemcpyAsync((CUdeviceptr) lower_indices_a, (CUdeviceptr) csr_lower_crows.data(), csr_lower_crows.shape(0)*sizeof(int), 0));
                   cuda_check(cuMemcpyAsync((CUdeviceptr) lower_indices_b, (CUdeviceptr) csr_lower_cols.data(), csr_lower_cols.shape(0)*sizeof(int), 0));

                   cuda_check(cuMemcpyAsync((CUdeviceptr) lower_value_indices, (CUdeviceptr) csr_lower_indices.data(), csr_lower_indices.shape(0)*sizeof(int), 0));

                   cuda_check(cuMemcpyAsync((CUdeviceptr) upper_indices_a, (CUdeviceptr) csc_upper_crows.data(), csc_upper_crows.shape(0)*sizeof(int), 0));
                   cuda_check(cuMemcpyAsync((CUdeviceptr) upper_indices_b, (CUdeviceptr) csc_upper_cols.data(), csc_upper_cols.shape(0)*sizeof(int), 0));

                   cuda_check(cuMemcpyAsync((CUdeviceptr) upper_value_indices, (CUdeviceptr) csc_upper_indices.data(), csc_upper_indices.shape(0)*sizeof(int), 0));

                   Float *raw_values_ptr = (Float *) malloc(raw_values.shape(0)*sizeof(Float));
                   cuda_check(cuMemcpyAsync((CUdeviceptr) raw_values_ptr, (CUdeviceptr) raw_values.data(), raw_values.shape(0)*sizeof(Float), 0));

                   new (self) Class(n_rows, csr_lower_cols.shape(0),
                                    lower_indices_a, lower_indices_b, lower_value_indices,
                                    upper_indices_a, upper_indices_b, upper_value_indices,
                                    raw_values.shape(0), raw_values_ptr,
                                    false);


                   free(lower_indices_a);
                   free(lower_indices_b);
                   free(lower_value_indices);
                   free(upper_indices_a);
                   free(upper_indices_b);
                   free(upper_value_indices);
                   free(raw_values_ptr);
                 } else if (csr_lower_crows.device_type() == nb::device::cpu::value) {
                   // CPU init
                   new (self) Class(n_rows, csr_lower_indices.shape(0),
                                    nullptr, nullptr, nullptr,
                                    nullptr, nullptr, nullptr,
                                    0, nullptr,
                                    true);
                 } else
                   throw std::invalid_argument("Unsupported input device! Only CPU and CUDA arrays are supported.");
               },
               nb::arg("n_rows"),
               nb::arg("csr_lower_crows"),
               nb::arg("csr_lower_cols"),
               nb::arg("csr_lower_indices"),
               nb::arg("csc_upper_crows"),
               nb::arg("csc_upper_cols"),
               nb::arg("csc_upper_indices"),
               nb::arg("raw_values"),
               doc_constructor)
          .def("debug_print", [](Class &self,
                                 nb::tensor<int, nb::c_contig> rows,
                                 nb::tensor<int, nb::c_contig> cols,
                                 nb::tensor<int, nb::c_contig> data_indices,
                                 nb::tensor<Float, nb::c_contig> raw_data,
                                 bool lower) {
                 scoped_set_context guard(cu_context);

                 self.debug_print(rows.data(), cols.data(), data_indices.data(), raw_data.data(), lower);
               },
               nb::arg("rows").noconvert(),
               nb::arg("cols").noconvert(),
               nb::arg("data_indices").noconvert(),
               nb::arg("raw_data").noconvert(),
               nb::arg("lower"),
               "Neill Debug Print Doc")
          .def("get_n_entries", [](Class &self) { return self.get_n_entries(); }, "Get NNZ")
          .def("get_n_rows", [](Class &self) { return self.get_n_rows(); }, "Get num rows")
          .def("get_n_raw_data", [](Class &self) { return self.get_n_raw_data(); }, "Get size of raw data array")
          .def("solve", [](Class &self,
                           nb::tensor<Float> raw_values,
                           nb::tensor<Float> b,
                           nb::tensor<Float> x,
                           bool skip_lower,
                           bool skip_upper){
                 if (skip_lower && skip_upper) {
                     throw std::invalid_argument("Cannot skip both lower and upper in solve.");
                 }

                 if (b.ndim() != 3) {
                   throw std::invalid_argument("Expected [NumBatch x NumPixels x NumSamples] tensor as input.");
                 }
                 if (x.ndim() != 3) {
                   throw std::invalid_argument("Expected [NumBatch x NumPixels x NumSamples] tensor as output.");
                 }

                 if (b.shape(0) != x.shape(0) || b.shape(1) != x.shape(1) || b.shape(2) != x.shape(2)) {
                   throw std::invalid_argument("Input and output dimensions should match.");
                 }

                 if (b.device_type() != x.device_type())
                   throw std::invalid_argument("x and b should be on the same device.");

                 if (raw_values.ndim() != 2) {
                   throw std::invalid_argument("raw_values needs to be [NumBatch x NumRawData].");
                 }
                 if (raw_values.shape(0) != b.shape(0))
                   throw std::invalid_argument("raw_values needs to have same batch dimensions.");

                 if (raw_values.shape(1) != self.get_n_raw_data())
                   throw std::invalid_argument("raw_values needs to have shape of get_n_raw_data().");

                 // CPU solve
                 if (b.device_type() == nb::device::cpu::value) {
                   throw std::invalid_argument("SupnCholeskySolver only defined for CUDA.");
                 }
                   // CUDA solve
                 else if (b.device_type() == nb::device::cuda::value) {
                   if (self.is_cpu())
                     throw std::invalid_argument("Input device is CUDA but the solver was initialized for CPU.");

                   scoped_set_context guard(cu_context);
                   int num_batch = b.shape(0);
                   // vdbg(num_batch)

                   for (int bIdx = 0; bIdx < num_batch; ++bIdx) {
                     //vdbg(bIdx);
                     //vdbg(raw_values.shape(1));
                     CUdeviceptr raw_values_ptr = raw_values.data() + bIdx * raw_values.shape(1);
                     CUdeviceptr b_ptr = b.data() + bIdx * (b.shape(1) * b.shape(2));
                     CUdeviceptr x_ptr = x.data() + bIdx * (x.shape(1) * x.shape(2));

                     self.solve_cuda(raw_values_ptr, b.shape(2),
                                     (CUdeviceptr) b_ptr, (CUdeviceptr) x_ptr,
                                     skip_lower, skip_upper);
//                     self.solve_cuda(raw_values.data(), b.ndim() == 2 ? b.shape(1) : 1,
//                                     (CUdeviceptr) b.data(), (CUdeviceptr) x.data(),
//                                     skip_lower, skip_upper);
                   }
                 }
                 else
                   throw std::invalid_argument("Unsupported input device! Only CPU and CUDA arrays are supported.");
               },
               nb::arg("raw_values").noconvert(),
               nb::arg("b").noconvert(),
               nb::arg("x").noconvert(),
               nb::arg("skip_lower"),
               nb::arg("skip_upper"),
               doc_solve);
}

NB_MODULE(_cholespy_core, m_) {
    (void) m_;

    nb::module_ m = nb::module_::import_("cholespy");

    nb::enum_<MatrixType>(m, "MatrixType", doc_matrix_type)
        .value("CSC", MatrixType::CSC)
        .value("CSR", MatrixType::CSR)
        .value("COO", MatrixType::COO);

    declare_cholesky<float>(m, "F", doc_cholesky_f);
    declare_cholesky<double>(m, "D", doc_cholesky_d);

    declare_supn_cholesky<float>(m, "F", doc_cholesky_f);
    declare_supn_cholesky<double>(m, "D", doc_cholesky_d);

    // Custom object to gracefully shutdown CUDA when unloading the module
    nb::detail::keep_alive(m.ptr(),
                           (void *) 1, // Unused payload
                           [](void *p) noexcept { shutdown_cuda(); });

    // Added by NDFC..
    m.def("inspect", [](nb::tensor<> tensor) {
        printf("Tensor data pointer : %p\n", tensor.data());
        printf("Tensor dimension : %zu\n", tensor.ndim());
        for (size_t i = 0; i < tensor.ndim(); ++i) {
            printf("Tensor dimension [%zu] : %zu\n", i, tensor.shape(i));
            printf("Tensor stride    [%zu] : %zd\n", i, tensor.stride(i));
        }
        printf("Device ID = %u (cpu=%i, cuda=%i)\n", tensor.device_id(),
            int(tensor.device_type() == nb::device::cpu::value),
            int(tensor.device_type() == nb::device::cuda::value)
        );
        printf("Tensor dtype: int16=%i, uint32=%i, float32=%i, float64=%i\n",
            tensor.dtype() == nb::dtype<int16_t>(),
            tensor.dtype() == nb::dtype<uint32_t>(),
            tensor.dtype() == nb::dtype<float>(),
            tensor.dtype() == nb::dtype<double>()
        );
    });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
