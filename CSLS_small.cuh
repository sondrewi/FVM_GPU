#ifndef CUDASLS_H
#define CUDASLS_H

#include <cuda_runtime.h>
#include <cusparse.h>
#include <chrono>
#include "SM_small.H"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

// Declare constant memory pointers
__constant__ double* entries_ptr;
__constant__ double* entries_diag_ptr;
__constant__ int* col_ptr;
__constant__ double* x_ptr;
__constant__ double* y_ptr;

//Check for CUDA Errors
#define CUDA_CHECK                                                  \
  {                                                                 \
    cudaDeviceSynchronize();                                        \
    cudaError_t err = cudaGetLastError();                           \
    if (err) {                                                      \
      std::cout << "Error: " << cudaGetErrorString(err) << " line " \
                << __LINE__ << " in CSLS_small.cu" << std::endl; \
      exit(1);                                                      \
    }                                                               \
  }


static __global__ void staircase_mult(int length, int row_start, int step_start, int n_rows);

static __global__ void add_kern(double* result, double* vec1, double* vec2, int length, bool minus = false, double* scale1 = nullptr, double* scale2 = nullptr);

static __global__ void dot_prod_kern(double* vec1, int length, double* reduc_arr, double* vec2);

static __global__ void ellpack_shared(double* diag, double* entries, int* col_idx, int* slice_ptr, double* x_dev, double* res, int n_rows);

static __global__ void smooth_kern(int algo, double* diag, double* entries, int* col_idx, int* slice_ptr, int* slice_mnz_arr, double* x_dev, double* b_dev, int n_rows, int max_iter, double tol, double* x_dev_new = nullptr);

static __global__ void calc_sums(double* diag, double* entries, int* col_idx, int* slice_ptr, double* D, double* R, int n_rows);

static __global__ void scale_sell(double* diag, double* entries, int* col_idx, int* slice_ptr, double* R, int n_rows);

static __global__ void scale_inv_sell(double* diag, double* entries, int* col_idx, int* slice_ptr, double* R, int n_rows);

static __global__ void block_reduce(double* read, double* write, int length, bool max = false);

static __device__ void warpReduce_sum(volatile double* sdata);

static __device__ void warpReduce_max(volatile double* sdata);

static __device__ void block_reduce_device(volatile double* smem, bool max = false);

static __global__ void Dinv_scale_kern(double* D, double* vec, int length);

static __global__ void D_scale_kern(double* D, double* vec, int length);

static __global__ void get_orig_res_kern(double* orig_res, double* D, double* res, int length);

static __global__ void max_deviation_kern(double* R, int length, double* reduc_arr);

static __global__ void ellpack_row_based(double* entries, int* col_idx, int* slice_ptr, double* x_dev, double* res, int n_rows);

static __device__ double CG_device(volatile double* smem, int* loc_nbrs, double* loc_nbrs_coef, double res_sq, double d_i, double diag_entry, int mnz, double tol, int max_iter);

static __device__ double Jacobi_device(volatile double* smem, int* loc_nbrs, double* loc_nbrs_coef, double b_i, double diag_entry, int mnz, double tol, int max_iter);

static __device__ double GS_device(volatile double* smem, int* loc_nbrs, double* loc_nbrs_coef, double b_i, double diag_entry, int mnz, double tol, int max_iter);

// Sparse Matrix class storing matrix either in
// compressed sparse column (csc) or compressed sparse row format
class CUDA_SLS {
 public:

    SparseMat mat;
    double* entries_dev = nullptr;
    double* diag_dev = nullptr;
    int* col_dev = nullptr;
    double* b_dev = nullptr;
    double* res_dev = nullptr;
    double* orig_res_dev = nullptr;
    double* d_dev = nullptr;
    double* Ad_dev = nullptr;
    double* dAd_dev = nullptr;
    double* rTr_dev = nullptr;
    double* nrTnr_dev = nullptr;
    double* x_dev = nullptr;
    double* D_dev = nullptr;
    double* R_dev = nullptr;
    double* R_max_dev = nullptr;
    double* R_max_CPU = nullptr;
    double* rTr_orig_dev = nullptr;
    double* rTr_orig_h = nullptr;
    double* x_dev_new = nullptr;

    double* reduc_arr1 = nullptr;
    double* reduc_arr2 = nullptr;

    int num_steps;
    cudaStream_t* streams;

    int* step_lengths = nullptr;
    int* row_starts = nullptr;
    int* step_starts = nullptr;
    int* step_blocks = nullptr;
    int n_rows;
    int slice_height;

    bool is_precond = false;

    size_t smooth_smem_size;

    int* slice_ptr_dev = nullptr;
    int* slice_mnz_dev = nullptr;

    std::string mult_algo;

    dim3 tile;
    dim3 blocks;

    dim3 vector_blocks;

    bool mult_initialised = false;

    CUDA_SLS(const std::string& file_name, const bool read_b, const bool sym_triang);

    void copy_system(std::string algo, double* b);

    void mult(double* result, double* vector);

    void CG_step();

    void reset_CG();

    void prepare_smoother(bool async);

    double calc_res();

    void reset_smoother();

    void reduce(double* result, bool max = false);

    void prepare_Ruiz_precond();

    int precond_Ruiz(double equi_tol = 1e-8, double check_frequency = 5);

    void dot_prod(double* result, double* vec1, double* vec2 = nullptr);

    void prepare_CG(double* x = nullptr, bool x_not_precond = true);

    double max_deviation();

    void diag_precond();

    void undo_precond();

    void smooth_sweeps(int algo = 0, int num_sweeps = 512, double tol = 1e-6, bool async = false);

    double get_orig_res_sq();

    void prepare_diag_precond();

    void scale_iter();

    void prepare_scaler();

    ~CUDA_SLS();
};

#endif

