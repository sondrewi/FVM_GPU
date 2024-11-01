#include "CSLS_small.cuh"

CUDA_SLS::CUDA_SLS(const std::string& file_name, const bool read_b, const bool sym_triang) : mat(file_name, read_b, sym_triang){
  //Constructor that constructs a SparseMatrix SM
}

void CUDA_SLS::mult(double* result, double* vector){
    if(mult_algo == "staircase"){
    //Launch kernels with 1024 threads in each to achieve maximal coalescence during read and write
    for (int i = 0; i < num_steps; ++i) {
        staircase_mult<<<step_blocks[i], slice_height, 0, streams[i]>>>(step_lengths[i], row_starts[i], step_starts[i], n_rows);
    }
    //cudaDeviceSynchronize();
    CUDA_CHECK
  }

  //Sliced Ellpack format
  else if(mult_algo == "ellpack_shared"){
    ellpack_shared<<<blocks, tile, tile.x*sizeof(double)>>>(diag_dev, entries_dev, col_dev, slice_ptr_dev, vector, result, n_rows);
    //CUDA_CHECK
  }

  else if(mult_algo == "ellpack_row_based"){
    ellpack_row_based<<<blocks, tile, tile.x*sizeof(double)>>>(entries_dev, col_dev, slice_ptr_dev, vector, result, n_rows);
    //CUDA_CHECK
  }
}

//Copy system of equations, i.e. matrix and RHS vector to GPU
//Get also the matrix storage format/multiplication algorithm
void CUDA_SLS::copy_system(std::string algo, double* b){
  n_rows = mat.n_rows;
  slice_height =  mat.slice_height;

  cudaMalloc((void**)&b_dev, sizeof(double) * n_rows);
  CUDA_CHECK
  cudaMemcpy(b_dev, b, sizeof(double) * n_rows, cudaMemcpyHostToDevice);
  CUDA_CHECK

  vector_blocks.x = (n_rows + 1023)/1024;
  cudaMalloc((void**)&reduc_arr1, vector_blocks.x*sizeof(double));
  CUDA_CHECK
  cudaMalloc((void**)&reduc_arr2, ((vector_blocks.x + 1023)/1024)*sizeof(double));
  CUDA_CHECK

  if(algo == "staircase"){
    num_steps = mat.nr_steps;
    streams = new cudaStream_t[num_steps];
    for (int i = 0; i < num_steps; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    mult_algo = "staircase";

    step_lengths = new int[num_steps];
    row_starts = new int[num_steps + 1]();
    step_starts = new int[num_steps + 1]();
    step_blocks = new int[num_steps];

    for(int i = 0; i < num_steps + 1; i++){
      if(i < num_steps){
        step_lengths[i] = mat.step_lengths[i];
        step_blocks[i] = mat.step_blocks[i];
      }

      row_starts[i] = mat.row_starts[i];
      step_starts[i] = mat.step_starts[i];
    }

    //allocate space
    cudaMalloc((void**)&entries_dev, sizeof(double) * mat.entries_size);
    CUDA_CHECK
    cudaMalloc((void**)&col_dev, sizeof(int) * mat.entries_size);
    CUDA_CHECK

    //Copy onto device
    cudaMemcpy(entries_dev, mat.entries, sizeof(double) * mat.entries_size,
           cudaMemcpyHostToDevice);
    CUDA_CHECK
    cudaMemcpy(col_dev, mat.col_idx, sizeof(int) * mat.entries_size,
           cudaMemcpyHostToDevice);
    CUDA_CHECK

    CUDA_CHECK(cudaMemcpyToSymbol(entries_ptr, &entries_dev, sizeof(double*)));
    CUDA_CHECK(cudaMemcpyToSymbol(col_ptr, &col_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(x_ptr, &d_dev, sizeof(double*)));
    CUDA_CHECK(cudaMemcpyToSymbol(y_ptr, &Ad_dev, sizeof(double*)));
  }

  else if(algo == "ellpack_shared"){
    tile.x = slice_height;
    blocks.x = (n_rows + slice_height - 1)/slice_height;

    mult_algo = "ellpack_shared";

    smooth_smem_size = slice_height*(sizeof(double) + mat.max_slice_mnz*(sizeof(int) + sizeof(double)));

    //allocate space
    cudaMalloc((void**)&entries_dev, sizeof(double) * mat.entries_size);
    CUDA_CHECK
    cudaMalloc((void**)&diag_dev, sizeof(double) * n_rows);
    CUDA_CHECK
    cudaMalloc((void**)&col_dev, sizeof(int) * mat.entries_size);
    CUDA_CHECK
    cudaMalloc((void**)&slice_ptr_dev, sizeof(int) * (mat.num_slices + 1));
    CUDA_CHECK
    cudaMalloc((void**)&slice_mnz_dev, sizeof(int) * mat.num_slices);
    CUDA_CHECK

    //Copy onto device
    cudaMemcpy(entries_dev, mat.entries, sizeof(double) * mat.entries_size,
           cudaMemcpyHostToDevice);
    CUDA_CHECK
    cudaMemcpy(diag_dev, mat.diag, sizeof(double) * mat.n_rows,
           cudaMemcpyHostToDevice);
    CUDA_CHECK
    cudaMemcpy(col_dev, mat.col_idx, sizeof(int) * mat.entries_size,
           cudaMemcpyHostToDevice);
    CUDA_CHECK
    cudaMemcpy(slice_ptr_dev, mat.sellp_slice_ptr, sizeof(int) * (mat.num_slices + 1),
           cudaMemcpyHostToDevice);
    CUDA_CHECK

    cudaMemcpy(slice_mnz_dev, mat.slice_mnz, sizeof(int) * mat.num_slices,
           cudaMemcpyHostToDevice);
    CUDA_CHECK
  }

  else if(algo == "ellpack_row_based"){
    tile.x = slice_height;
    blocks.x = (n_rows + slice_height - 1)/slice_height;

    mult_algo = "ellpack_row_based";

    //allocate space
    cudaMalloc((void**)&entries_dev, sizeof(double) * mat.entries_size);
    CUDA_CHECK
    cudaMalloc((void**)&col_dev, sizeof(int) * mat.entries_size);
    CUDA_CHECK
    cudaMalloc((void**)&slice_ptr_dev, sizeof(int) * (mat.num_slices + 1));
    CUDA_CHECK

    //Copy onto device
    cudaMemcpy(entries_dev, mat.entries, sizeof(double) * mat.entries_size,
           cudaMemcpyHostToDevice);
    CUDA_CHECK
    cudaMemcpy(col_dev, mat.col_idx, sizeof(int) * mat.entries_size,
           cudaMemcpyHostToDevice);
    CUDA_CHECK
    cudaMemcpy(slice_ptr_dev, mat.sellp_slice_ptr, sizeof(int) * (mat.num_slices + 1),
           cudaMemcpyHostToDevice);
    CUDA_CHECK
  }
}

double CUDA_SLS::calc_res(){
  mult(Ad_dev, x_dev);
  add_kern<<<vector_blocks.x, 1024>>>(Ad_dev, b_dev, Ad_dev, n_rows, true);
  //CUDA_CHECK

  dot_prod(rTr_dev, Ad_dev);
  //CUDA_CHECK

  double* rTr_h = new double[1];

  cudaMemcpy(rTr_h, rTr_dev, sizeof(double),cudaMemcpyDeviceToHost);
  //CUDA_CHECK

  double rTr = rTr_h[0];
  delete[] rTr_h;

  return rTr;
}

void CUDA_SLS::prepare_smoother(bool async){
  if(x_dev == nullptr){
    cudaMalloc((void**)&x_dev, n_rows*sizeof(double));
    CUDA_CHECK
  }

  if(Ad_dev == nullptr){
    cudaMalloc((void**)&Ad_dev, sizeof(double)*n_rows);
    CUDA_CHECK
  }

  if(rTr_dev == nullptr){
    cudaMalloc((void**)&rTr_dev, sizeof(double));
    CUDA_CHECK
  }

  if(!async && x_dev_new == nullptr){
    cudaMalloc((void**)&x_dev_new, n_rows*sizeof(double));
    CUDA_CHECK
  }

  dot_prod(rTr_dev, b_dev);
  CUDA_CHECK
}

void CUDA_SLS::reset_smoother(){
  cudaMemset(x_dev, 0.0, sizeof(double)*n_rows);
}

//Make a Conjugate-Gradient Step on GPU
void CUDA_SLS::CG_step(){
  mult(Ad_dev, d_dev);
  //CUDA_CHECK

  dot_prod(dAd_dev, d_dev, Ad_dev);
  //CUDA_CHECK

  add_kern<<<blocks, tile>>>(x_dev, x_dev, d_dev, n_rows, false, rTr_dev, dAd_dev);
  //CUDA_CHECK

  add_kern<<<blocks, tile>>>(res_dev, res_dev, Ad_dev, n_rows, true, rTr_dev, dAd_dev);
  //CUDA_CHECK

  dot_prod(nrTnr_dev, res_dev);
  //CUDA_CHECK
  add_kern<<<blocks, tile>>>(d_dev, res_dev, d_dev, n_rows, false, nrTnr_dev, rTr_dev);
  //CUDA_CHECK

  std::swap(nrTnr_dev, rTr_dev);
}

//Recompute residual and set b and d to this residual. Set x to zero
void CUDA_SLS::reset_CG(){
    mult(Ad_dev, x_dev);
    CUDA_CHECK
    add_kern<<<vector_blocks, 1024>>>(res_dev, b_dev, Ad_dev, n_rows, true);
    CUDA_CHECK
    cudaMemcpy(b_dev, res_dev, sizeof(double)*n_rows, cudaMemcpyDeviceToDevice);
    CUDA_CHECK
    cudaMemcpy(d_dev, res_dev, sizeof(double)*n_rows, cudaMemcpyDeviceToDevice);
    CUDA_CHECK
    cudaMemset(x_dev, 0.0, sizeof(double)*n_rows);
    CUDA_CHECK
    dot_prod(rTr_dev, res_dev);
    CUDA_CHECK
}

void CUDA_SLS::prepare_CG(double* x, bool x_not_precond){
  if(d_dev == nullptr){
    cudaMalloc((void**)&d_dev, sizeof(double)*n_rows);
    CUDA_CHECK
  }

  if(res_dev == nullptr){
    cudaMalloc((void**)&res_dev, sizeof(double)*n_rows);
    CUDA_CHECK
  }

  if(Ad_dev == nullptr){
    cudaMalloc((void**)&Ad_dev, sizeof(double)*n_rows);
    CUDA_CHECK
  }

  if(dAd_dev == nullptr){
    cudaMalloc((void**)&dAd_dev, sizeof(double));
    CUDA_CHECK
  }

  if(rTr_dev == nullptr){
    cudaMalloc((void**)&rTr_dev, sizeof(double));
    CUDA_CHECK
  }

  if(nrTnr_dev == nullptr){
    cudaMalloc((void**)&nrTnr_dev, sizeof(double));
    CUDA_CHECK
  }

  if(x_dev == nullptr){
    cudaMalloc((void**)&x_dev, sizeof(double)*n_rows);
    CUDA_CHECK
  }

  //If x-guess, x' given, set b -= Ax'
  if(x!=nullptr){
    cudaMemcpy(x_dev, x, sizeof(double)*n_rows, cudaMemcpyHostToDevice);
    CUDA_CHECK
    if(x_not_precond && is_precond){
      D_scale_kern<<<vector_blocks, 1024>>>(D_dev, x_dev, n_rows);
      CUDA_CHECK
    }

    reset_CG();
    CUDA_CHECK
  }

  else {
    cudaMemcpy(res_dev, b_dev, sizeof(double)*n_rows, cudaMemcpyDeviceToDevice);
    CUDA_CHECK
    cudaMemcpy(d_dev, b_dev, sizeof(double)*n_rows, cudaMemcpyDeviceToDevice);
    CUDA_CHECK
    cudaMemset(x_dev, 0.0, sizeof(double)*n_rows);
    CUDA_CHECK
    dot_prod(rTr_dev, res_dev);
    CUDA_CHECK
  }
}

void CUDA_SLS::prepare_diag_precond(){
  if(rTr_orig_dev == nullptr){
    cudaMalloc((void**)&rTr_orig_dev, sizeof(double));
    CUDA_CHECK
    rTr_orig_h = new double[1];
  }

  if(D_dev == nullptr){
    cudaMalloc((void**)&D_dev, sizeof(double)*n_rows);
    CUDA_CHECK
  }

  if(orig_res_dev == nullptr){
    cudaMalloc((void**)&orig_res_dev, sizeof(double)*n_rows);
    CUDA_CHECK
  }
}

void CUDA_SLS::diag_precond(){
  is_precond = true;

  cudaMemcpy(D_dev, diag_dev, n_rows*sizeof(double), cudaMemcpyDeviceToDevice);
  //CUDA_CHECK

  Dinv_scale_kern<<<vector_blocks, 1024>>>(D_dev, b_dev, n_rows);
  //CUDA_CHECK
  scale_sell<<<blocks, tile, sizeof(double)*tile.x>>>(diag_dev, entries_dev, col_dev, slice_ptr_dev, D_dev, n_rows);
  //CUDA_CHECK
}

void CUDA_SLS::prepare_Ruiz_precond(){
  is_precond = true;

  if(D_dev == nullptr){
    cudaMalloc((void**)&D_dev, sizeof(double)*n_rows);
    CUDA_CHECK
  }

  if(orig_res_dev == nullptr){
    cudaMalloc((void**)&orig_res_dev, sizeof(double)*n_rows);
    CUDA_CHECK
  }

  double* D_h = new double[n_rows];
  std::fill(D_h, D_h + n_rows, 1.0);

  cudaMemcpy(D_dev, D_h, n_rows*sizeof(double), cudaMemcpyHostToDevice);
  CUDA_CHECK

  delete[] D_h;

  R_max_CPU = new double[1];

  if(R_max_dev == nullptr){
    cudaMalloc((void**)&R_max_dev, sizeof(double));
    CUDA_CHECK
  }

  if(R_dev == nullptr){
    cudaMalloc((void**)&R_dev, sizeof(double)*n_rows);
    CUDA_CHECK
  }

  if(rTr_orig_dev == nullptr){
    cudaMalloc((void**)&rTr_orig_dev, sizeof(double));
    CUDA_CHECK
    rTr_orig_h = new double[1];
  }
}

int CUDA_SLS::precond_Ruiz(double equi_tol, double check_frequency){
  prepare_Ruiz_precond();
  double R_max_deviation = 2;
  int num_iters = 0;

  while(R_max_deviation > equi_tol){
    for(int i = 0; i < check_frequency; i++){
      scale_iter();
      //CUDA_CHECK
      num_iters +=1;
    }

    R_max_deviation = max_deviation();
  }

  Dinv_scale_kern<<<vector_blocks, 1024>>>(D_dev, b_dev, n_rows);

  //CUDA_CHECK
  return num_iters;
}

double CUDA_SLS::max_deviation(){
  max_deviation_kern<<<vector_blocks, 1024>>>(R_dev, n_rows, reduc_arr1);
  reduce(R_max_dev, true);
  cudaMemcpy(R_max_CPU, R_max_dev, sizeof(double), cudaMemcpyDeviceToHost);
  return R_max_CPU[0];
}

void CUDA_SLS::undo_precond(){
  if(is_precond){
    scale_inv_sell<<<blocks, tile, sizeof(double)*tile.x>>>(diag_dev, entries_dev, col_dev, slice_ptr_dev, D_dev, n_rows);
    D_scale_kern<<<vector_blocks, 1024>>>(D_dev, b_dev, n_rows);
  }
  is_precond = false;
}

CUDA_SLS::~CUDA_SLS(){
    if(entries_dev != nullptr){
      cudaFree(entries_dev);
      CUDA_CHECK
    }

    if(col_dev != nullptr){
      cudaFree(col_dev);
      CUDA_CHECK
    }

    if(res_dev != nullptr){
      cudaFree(res_dev);
      CUDA_CHECK
    }

    if(d_dev != nullptr){
      cudaFree(d_dev);
      CUDA_CHECK
    }

    if(Ad_dev != nullptr){
      cudaFree(Ad_dev);
      CUDA_CHECK
    }

    if(dAd_dev != nullptr){
      cudaFree(dAd_dev);
      CUDA_CHECK
    }

    if(rTr_dev != nullptr){
      cudaFree(rTr_dev);
      CUDA_CHECK
    }

    if(nrTnr_dev != nullptr){
      cudaFree(nrTnr_dev);
      CUDA_CHECK
    }

    if(x_dev != nullptr){
      cudaFree(x_dev);
      CUDA_CHECK
    }

    if(slice_ptr_dev != nullptr){
      cudaFree(slice_ptr_dev);
      CUDA_CHECK
    }

    if(step_lengths != nullptr){
      cudaFree(step_lengths);
      CUDA_CHECK
    }

    if(row_starts != nullptr){
      cudaFree(row_starts);
      CUDA_CHECK
    }

    if(step_starts != nullptr){
      cudaFree(step_starts);
      CUDA_CHECK
    }

    if(step_blocks != nullptr){
      cudaFree(step_blocks);
      CUDA_CHECK
    }

    if(mult_algo == "staircase"){
      for(int i = 0; i < num_steps; i++){
        cudaStreamDestroy(streams[i]);
        CUDA_CHECK
      }
    }
}

void CUDA_SLS::smooth_sweeps(int algo, int num_sweeps, double tol, bool async){
  if(!async){
    cudaFuncSetAttribute(smooth_kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smooth_smem_size);
    smooth_kern<<<blocks, tile, smooth_smem_size>>>(algo, diag_dev, entries_dev, col_dev, slice_ptr_dev, slice_mnz_dev, x_dev, b_dev, n_rows, num_sweeps, tol, x_dev_new);
    //CUDA_CHECK

    std::swap(x_dev_new, x_dev);
  }

  else{
    cudaFuncSetAttribute(smooth_kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smooth_smem_size);
    smooth_kern<<<blocks, tile, smooth_smem_size>>>(algo, diag_dev, entries_dev, col_dev, slice_ptr_dev, slice_mnz_dev, x_dev, b_dev, n_rows, num_sweeps, tol);
    //CUDA_CHECK
  }

}

void CUDA_SLS::scale_iter(){
  calc_sums<<<blocks, tile>>>(diag_dev, entries_dev, col_dev, slice_ptr_dev, D_dev, R_dev, n_rows);
  //CUDA_CHECK

  scale_sell<<<blocks, tile, sizeof(double)*tile.x>>>(diag_dev, entries_dev, col_dev, slice_ptr_dev, R_dev, n_rows);
  //CUDA_CHECK
}

double CUDA_SLS::get_orig_res_sq(){
  cudaMemcpy(orig_res_dev, res_dev, n_rows*sizeof(double), cudaMemcpyDeviceToDevice);

  D_scale_kern<<<vector_blocks, 1024>>>(D_dev, orig_res_dev, n_rows);
  //CUDA_CHECK

  dot_prod(rTr_orig_dev, orig_res_dev);
  //CUDA_CHECK

  cudaMemcpy(rTr_orig_h, rTr_orig_dev, sizeof(double), cudaMemcpyDeviceToHost);
  //CUDA_CHECK

  return rTr_orig_h[0];
}

void CUDA_SLS::dot_prod(double* result, double* vec1, double* vec2){
  dot_prod_kern<<<vector_blocks, 1024>>>(vec1, n_rows, reduc_arr1, vec2);
  //CUDA_CHECK
  reduce(result);
  //CUDA_CHECK
}

void CUDA_SLS::reduce(double* result, bool max){
    //Initial length is just the number of blocks needed to reduce vector of n_rows entries
    //and this number is stored in vector_block.x
    int length = vector_blocks.x;

    //In turn, the number of blocks we need to reduce a vector of length vector_blocks.x is calculated
    int num_blocks = (length + 1023)/1024;


    if(!max){
      if(length > 1024){
          block_reduce<<<num_blocks, 1024>>>(reduc_arr1, reduc_arr2, length);
          length = num_blocks;
          block_reduce<<<1, 1024>>>(reduc_arr2, result, length);
      }

      else{
        block_reduce<<<1, 1024>>>(reduc_arr1, result, length);
      }
    }

    else{
      if(length > 1024){
          block_reduce<<<num_blocks, 1024>>>(reduc_arr1, reduc_arr2, length, true);
          length = num_blocks;
          block_reduce<<<1, 1024>>>(reduc_arr2, result, length, true);
      }

      else{
        block_reduce<<<1, 1024>>>(reduc_arr1, result, length, true);
      }
    }
}

__global__ void staircase_mult(int length, int row_start, int step_start, int n_rows){
  int read_idx = step_start + length*blockDim.x*blockIdx.x + threadIdx.x;
  int write_idx = row_start + blockIdx.x*blockDim.x + threadIdx.x;

  double row_sum = 0;

  double entry;
  int col_id;
  double x_i;
  for(int i = 0; i < length; i++){
      col_id = col_ptr[read_idx];
      entry = entries_ptr[read_idx];
      if(col_id > 0){
        x_i = x_ptr[col_id];
        row_sum += entry*x_i;
      }
      read_idx += blockDim.x;
  }

  if(write_idx < n_rows){
    y_ptr[write_idx] = row_sum;
  }
}

//Sliced ELLPACK row-vertical storage
__global__ void ellpack_shared(double* diag, double* entries, int* col_idx, int* slice_ptr, double* x_dev, double* res, int n_rows){
  extern __shared__ double x_diag[];

  int read_start = slice_ptr[blockIdx.x];
  int read_end = slice_ptr[blockIdx.x + 1];
  int num_cols = (read_end - read_start)/blockDim.x;
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int read_idx = read_start + threadIdx.x;
  bool in_domain = (row < n_rows);

  int col_id;
  double entry = (in_domain) ? diag[row] : 0;
  double x_i = (in_domain) ? x_dev[row]: 0;

  x_diag[threadIdx.x] = x_i;
  double row_sum = x_i*entry;

  __syncthreads();

  for(int i = 0; i < num_cols; i++){
    col_id = col_idx[read_idx];
    entry = entries[read_idx];

    if(col_id > 0){
      x_i = x_dev[col_id - 1];
    }

    else if(col_id < 0){
      x_i = x_diag[-(col_id + 1)];
    }

    row_sum += entry*x_i;
    read_idx += blockDim.x;

    __syncwarp();
  }

  if(in_domain){
    res[row] = row_sum;
  }
}

__global__ void max_deviation_kern(double* R, int length, double* reduc_arr){
  __shared__ double smem[1024];
  int gl_id = blockIdx.x*blockDim.x + threadIdx.x;

  if(gl_id < length){
    smem[threadIdx.x] = fabs(1 - R[gl_id]);
  }

  else{
    smem[threadIdx.x] = 0;
  }

  block_reduce_device(smem, true);

  if(threadIdx.x == 0){
    reduc_arr[blockIdx.x] = smem[0];
  }
}

//Dot product kernel (process 1024 entries in a thread-block)
__global__ void dot_prod_kern(double* vec1, int length, double* reduc_arr, double* vec2){
  __shared__ double smem[1024];
  int gl_id = blockIdx.x*blockDim.x + threadIdx.x;

  if(gl_id < length){
    double x_i = vec1[gl_id];
    double y_i = (vec2 != nullptr) ? vec2[gl_id] : x_i;
    smem[threadIdx.x] = x_i*y_i;
  }

  else{
    smem[threadIdx.x] = 0;
  }

  block_reduce_device(smem);

  if(threadIdx.x == 0){
    reduc_arr[blockIdx.x] = smem[0];
  }
}

__device__ void warpReduce_sum(volatile double* sdata){
    double vsum = sdata[threadIdx.x] + sdata[threadIdx.x+32];
    vsum += __shfl_down_sync(0xffffffff, vsum, 16);
    vsum += __shfl_down_sync(0xffffffff, vsum, 8);
    vsum += __shfl_down_sync(0xffffffff, vsum, 4);
    vsum += __shfl_down_sync(0xffffffff, vsum, 2);
    vsum += __shfl_down_sync(0xffffffff, vsum, 1);

    if (threadIdx.x == 0) {
      sdata[0] = vsum;
    }
}

__device__ void warpReduce_max(volatile double* sdata){
    double vmax = fmax(sdata[threadIdx.x], sdata[threadIdx.x+32]);
    vmax += fmax(vmax, __shfl_down_sync(0xffffffff, vmax, 16));
    vmax += fmax(vmax, __shfl_down_sync(0xffffffff, vmax, 8));
    vmax += fmax(vmax, __shfl_down_sync(0xffffffff, vmax, 4));
    vmax += fmax(vmax, __shfl_down_sync(0xffffffff, vmax, 2));
    vmax += fmax(vmax, __shfl_down_sync(0xffffffff, vmax, 1));

    if (threadIdx.x == 0) {
      sdata[0] = vmax;
    }
}

__global__ void block_reduce(double* read, double* write, int length, bool max){
    __shared__ double smem[1024];
    int gl_id = blockIdx.x*1024 + threadIdx.x;
    smem[threadIdx.x] = (gl_id < length) ? read[gl_id] : 0;

    block_reduce_device(smem, max);

    if (threadIdx.x == 0) {
      write[blockIdx.x] = smem[0];
    }
}

__device__ void block_reduce_device(volatile double* smem, bool max){
    if(max){
      __syncthreads();
      if(threadIdx.x < 512 && blockDim.x > 512){
        smem[threadIdx.x] = fmax(smem[threadIdx.x], smem[threadIdx.x + 512]);
      }
      __syncthreads();
      if(threadIdx.x < 256 && blockDim.x > 256){
        smem[threadIdx.x] = fmax(smem[threadIdx.x], smem[threadIdx.x + 256]);
      }
      __syncthreads();
      if(threadIdx.x < 128 && blockDim.x > 128){
        smem[threadIdx.x] = fmax(smem[threadIdx.x], smem[threadIdx.x + 128]);
      }
      __syncthreads();
      if(threadIdx.x < 64 && blockDim.x > 64){
        smem[threadIdx.x] = fmax(smem[threadIdx.x], smem[threadIdx.x + 64]);
      }
      __syncthreads();
      if (threadIdx.x < 32) {
        warpReduce_max(smem);
      }
      __syncthreads();
    }

    else{
      __syncthreads();
      if(threadIdx.x < 512 && blockDim.x > 512){
        smem[threadIdx.x] += smem[threadIdx.x + 512];
      }
      __syncthreads();
      if(threadIdx.x < 256 && blockDim.x > 256){
        smem[threadIdx.x] += smem[threadIdx.x + 256];
      }
      __syncthreads();
      if(threadIdx.x < 128 && blockDim.x > 128){
        smem[threadIdx.x] += smem[threadIdx.x + 128];
      }
      __syncthreads();
      if(threadIdx.x < 64 && blockDim.x > 64){
        smem[threadIdx.x] += smem[threadIdx.x + 64];
      }
      __syncthreads();
      if (threadIdx.x < 32) {
        warpReduce_sum(smem);
      }
      __syncthreads();
    }
}

//Add two vectors
__global__ void add_kern(double* result, double* vec1, double* vec2, int length, bool minus, double* scale1, double* scale2){
  double num = (scale1 != nullptr) ? scale1[0] : 1;
  double denom = (scale2 != nullptr) ? scale2[0] : 1;
  int gl_id = blockIdx.x*blockDim.x + threadIdx.x;

  double scale = (1 - 2 * minus)*num/denom;

  __syncthreads();

  if(gl_id < length){
    double vec1_i = vec1[gl_id];
    double vec2_i = scale*vec2[gl_id];
    result[gl_id] = vec1_i + vec2_i;
  }
}

__global__ void Dinv_scale_kern(double* D, double* vec, int length){
  int gl_id = blockIdx.x*1024 + threadIdx.x;
  if(gl_id < length){
    double D_i = D[gl_id];
    vec[gl_id] /= sqrt(D_i);
  }
}

__global__ void D_scale_kern(double* D, double* vec, int length){
  int gl_id = blockIdx.x*1024 + threadIdx.x;
  if(gl_id < length){
    double D_i = D[gl_id];
    vec[gl_id] *= sqrt(D_i);
  }
}

__global__ void smooth_kern(int algo, double* diag, double* entries, int* col_idx, int* slice_ptr, int* slice_mnz_arr, double* x_dev, double* b_dev, int n_rows, int max_iter, double tol, double* x_dev_new){
  extern __shared__ char smem_u[];

  int read_start = slice_ptr[blockIdx.x];
  int read_end = slice_ptr[blockIdx.x + 1];
  int mnz = slice_mnz_arr[blockIdx.x];
  int num_cols = (read_end - read_start)/blockDim.x;
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int read_idx = read_start + threadIdx.x;
  int write_idx = threadIdx.x;

  double* smem = reinterpret_cast<double*>(smem_u);
  int* loc_nbrs = reinterpret_cast<int*>(smem_u + blockDim.x * sizeof(double));
  double* loc_nbrs_coef = reinterpret_cast<double*>(smem_u + blockDim.x *(sizeof(double) + mnz * sizeof(int)));
  bool in_domain = (row < n_rows);

  double x_i = (in_domain) ? x_dev[row] : 0;
  double diag_entry = (in_domain) ? diag[row] : 1;
  double b_i = (in_domain) ? b_dev[row] : 0;

  //Find first residual square norm

  b_i -= diag_entry*x_i;

  smem[threadIdx.x] = x_i;

  //Prepare SELL arrays for local block matrix
  for(int i = 0; i < mnz; i++){
    loc_nbrs[i*blockDim.x + threadIdx.x] = -1;
    loc_nbrs_coef[i*blockDim.x + threadIdx.x] = 0;
  }

  __syncthreads();

  //Get RHS of local problem
  for(int i = 0; i < num_cols; i++){
    int col_id = col_idx[read_idx];
    double entry = entries[read_idx];

    //Check if col idx is outside block
    if(col_id > 0){
      double x_coef = x_dev[col_id-1];
      b_i -= entry*x_coef;
    }

    //If it is in local block, add index and coef
    else if (col_id != 0){
      loc_nbrs[write_idx] = -(col_id+1);
      loc_nbrs_coef[write_idx] = entry;
      b_i -= smem[-(col_id+1)]*entry;
      write_idx += blockDim.x;
    }

    read_idx += blockDim.x;
  }

  __syncthreads();

  //Find first residual square norm
  smem[threadIdx.x] = b_i*b_i;
  block_reduce_device(smem);
  double res_sq = smem[0];

  double y_i = 0;

  __syncthreads();

  if(algo == 0 && res_sq > tol){
    y_i = CG_device(smem, loc_nbrs, loc_nbrs_coef, res_sq, b_i, diag_entry, mnz, tol, max_iter);
  }

  else if(algo == 1 && res_sq > tol){
    y_i = Jacobi_device(smem, loc_nbrs, loc_nbrs_coef, b_i, diag_entry, mnz, tol, max_iter);
  }

  else if(algo == 2 && res_sq > tol){
    y_i = GS_device(smem, loc_nbrs, loc_nbrs_coef, b_i, diag_entry, mnz, tol, max_iter);
  }

  if(x_dev_new == nullptr){
    if(in_domain){
      x_dev[row] = y_i + x_i;
    }

    __threadfence_system();
  }

  else{
    if(in_domain){
      x_dev_new[row] = y_i + x_i;
    }
  }
}

__device__ double CG_device(volatile double* smem, int* loc_nbrs, double* loc_nbrs_coef, double res_sq, double d_i, double diag_entry, int mnz, double tol, int max_iter){
    int iter_count = 0;
    double res_i = d_i;
    double x_i = 0;

     while(iter_count < max_iter){
      smem[threadIdx.x] = d_i;
      __syncthreads();
      //Compute matrix-vector product
      double Ad_i = diag_entry*d_i;
      int read_idx = threadIdx.x;
      for(int i = 0; i < mnz; i++){
        int col_id = loc_nbrs[read_idx];

        if(col_id >= 0){
          Ad_i += loc_nbrs_coef[read_idx]*smem[col_id];
        }
        read_idx += blockDim.x;
      }
      __syncthreads();

      smem[threadIdx.x] = Ad_i*d_i;
      block_reduce_device(smem);

      double alpha = res_sq/smem[0];
      x_i += alpha*d_i;
      res_i -= alpha*Ad_i;

      __syncthreads();
      smem[threadIdx.x] = res_i*res_i;
      block_reduce_device(smem);
      double new_res_sq = smem[0];
      __syncthreads();

      if(new_res_sq < tol){
        return x_i;
      }

      d_i = res_i + (new_res_sq/res_sq)*d_i;
      res_sq = new_res_sq;
      iter_count += 1;
    }

    return x_i;
}

__device__ double Jacobi_device(volatile double* smem, int* loc_nbrs, double* loc_nbrs_coef, double b_i, double diag_entry, int mnz, double tol, int max_iter){
  int iter_count = 0;
  double x_i = b_i / diag_entry;

  while(iter_count < max_iter){
    smem[threadIdx.x] = x_i;
    __syncthreads();
    int read_idx = threadIdx.x;
    double res_i = b_i - x_i*diag_entry;
    for(int i = 0; i < mnz; i++){
      int col_id = loc_nbrs[read_idx];
      if(col_id >= 0){
        res_i -= loc_nbrs_coef[read_idx]*smem[col_id];
      }
      read_idx += blockDim.x;
    }

    __syncthreads();
    smem[threadIdx.x] = res_i*res_i;
    block_reduce_device(smem);
    double res_sq = smem[0];

    __syncthreads();

    if(res_sq < tol){
      return x_i;
    }

    x_i += res_i/diag_entry;
    iter_count += 1;
  }

  return x_i;
}

__device__ double GS_device(volatile double* smem, int* loc_nbrs, double* loc_nbrs_coef, double b_i, double diag_entry, int mnz, double tol, int max_iter){
  int iter_count = 0;
  double x_i = 0;
  int num_warps = blockDim.x/32;

  while(iter_count < max_iter){
    int read_idx = threadIdx.x;
    __syncthreads();
    smem[threadIdx.x] = x_i;
    __syncthreads();
    for(int i = 0; i < num_warps; i++){
      if(threadIdx.x/32 == i){
        double sum = b_i;
        for(int j = 0; j < mnz; j++){
          int col_id = loc_nbrs[read_idx];

          if(col_id >= 0){
            sum -= loc_nbrs_coef[read_idx]*smem[col_id];
          }
          read_idx += blockDim.x;
        }
        x_i = sum/diag_entry;
        smem[threadIdx.x] = x_i;
      }
      __syncthreads();
    }

    double res_i = b_i - x_i*diag_entry;
    read_idx = threadIdx.x;
    for(int j = 0; j < mnz; j++){
      int col_id = loc_nbrs[read_idx];

      if(col_id >= 0){
        res_i -= loc_nbrs_coef[read_idx]*smem[col_id];
      }
      read_idx += blockDim.x;
    }

    __syncthreads();

    smem[threadIdx.x] = res_i*res_i;
    block_reduce_device(smem);
    double res_sq = smem[0];
    __syncthreads();


    if(res_sq < tol){
      return x_i;
    }

    iter_count += 1;
  }

  return x_i;
}

__global__ void calc_sums(double* diag, double* entries, int* col_idx, int* slice_ptr, double* D, double* R, int n_rows){
  int read_start = slice_ptr[blockIdx.x];
  int read_end = slice_ptr[blockIdx.x + 1];
  int num_cols = (read_end - read_start)/blockDim.x;
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int read_idx = read_start + threadIdx.x;

  double entry = diag[row];
  double row_norm = entry*entry;

  for(int i = 0; i < num_cols; i++){
    entry = entries[read_idx];
    row_norm += entry*entry;
    read_idx += blockDim.x;
  }

  __syncthreads();

  if(row < n_rows){
    double scaler = sqrt(row_norm);
    R[row] = scaler;
    D[row] *= scaler;
  }
}

__global__ void scale_sell(double* diag, double* entries, int* col_idx, int* slice_ptr, double* R, int n_rows){
  extern __shared__ double R_local[];

  int read_start = slice_ptr[blockIdx.x];
  int read_end = slice_ptr[blockIdx.x + 1];
  int num_cols = (read_end - read_start)/blockDim.x;
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int read_idx = read_start + threadIdx.x;

  double R_i, R_row;
  int col_id;
  if(row < n_rows){
    R_i = sqrt(R[row]);
    R_row = R_i;
    R_local[threadIdx.x] = R_i;
    diag[row] /= R_i*R_i;
  }

  for(int i = 0; i < num_cols; i++){
    __syncwarp();

    col_id = col_idx[read_idx];

    if(col_id > 0){
      R_i = sqrt(R[col_id - 1]);
    }

    else if(col_id < 0){
      R_i = R_local[-(col_id + 1)];
    }

    entries[read_idx] /= R_i*R_row;
    read_idx += blockDim.x;
  }
}

__global__ void scale_inv_sell(double* diag, double* entries, int* col_idx, int* slice_ptr, double* R, int n_rows){
  extern __shared__ double R_local[];

  int read_start = slice_ptr[blockIdx.x];
  int read_end = slice_ptr[blockIdx.x + 1];
  int num_cols = (read_end - read_start)/blockDim.x;
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int read_idx = read_start + threadIdx.x;

  double R_i, R_row;
  int col_id;
  if(row < n_rows){
    R_i = sqrt(R[row]);
    R_row = R_i;
    R_local[threadIdx.x] = R_i;
    diag[row] *= R_i*R_i;
  }

  for(int i = 0; i < num_cols; i++){
    __syncwarp();

    col_id = col_idx[read_idx];

    if(col_id > 0){
      R_i = sqrt(R[col_id - 1]);
    }

    else if(col_id < 0){
      R_i = R_local[-(col_id + 1)];
    }

    entries[read_idx] *= R_i*R_row;
    read_idx += blockDim.x;
  }
}

__global__ void Chol_sell(double* A, double* A_diag, double* L, double* L_diag, int* col_idx, int* slice_ptr, double* res, int n_rows){
  int read_start = slice_ptr[blockIdx.x];
  int read_end = slice_ptr[blockIdx.x + 1];
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int read_start_i = read_start + threadIdx.x;
  int rw_idx = read_start_i;
  int num_cols = (read_end - read_start)/blockDim.x;

  //get current diagonal on this row
  double L_ii = L_diag[i];

  //iterate over non-zero elements in row i
  for (int alpha = 0; alpha < num_cols; alpha++){
    __syncthreads();

    int j = col_idx[rw_idx];

    if(j < 0){
      continue;
    }

    //Get the slice and starting index of row j
    int slice = j/blockDim.x;
    int slice_start = slice_ptr[slice];
    int slice_end = slice_ptr[slice + 1];
    int num_cols_j = (slice_end - slice_start)/blockDim.x;
    int read_start_j = slice_start + j%blockDim.x;

    //Initiate s to a_ij
    double s = A[rw_idx];

    //iterate over the non-zero elements in row j
    for(int beta = 0; beta < num_cols_j; beta++){
      int j_entry_col = col_idx[read_start_j + beta*blockDim.x];

      //check against available non-zero column indices in row i
      for (int gamma = 0; gamma < alpha; gamma++){
        int i_entry_col = col_idx[read_start_i + gamma*blockDim.x];

        if(j_entry_col == i_entry_col){
          s -= L[read_start_i + gamma*blockDim.x]*L[read_start_j + beta*blockDim.x];
        }

        else if(j_entry_col < i_entry_col){
          break;
        }
      }
    }

    L[rw_idx] = s/L_ii;
    rw_idx += blockDim.x;
  }


  L_ii = A_diag[i];

  for (int k = 0; k < num_cols; k++){
    L_ii -= pow(L[read_start_i + k*blockDim.x],2);
  }

  L_diag[i] = sqrt(L_ii);
}

__global__ void ellpack_row_based(double* entries, int* col_idx, int* slice_ptr, double* x_dev, double* res, int n_rows){
  int read_start = slice_ptr[blockIdx.x];
  int read_end = slice_ptr[blockIdx.x + 1];
  int num_cols = (read_end - read_start)/blockDim.x;
  int write_idx = blockIdx.x*blockDim.x + threadIdx.x;
  int read_idx = read_start + threadIdx.x;

  double row_sum = 0;

  for(int i = 0; i < num_cols; i++){
    int col_id = col_idx[read_idx];
    double entry = entries[read_idx];

    if(col_id >= 0){
      double x_i = x_dev[col_id];
      row_sum += entry*x_i;
    }

    read_idx += blockDim.x;
  }

  if(write_idx < n_rows){
    res[blockIdx.x*blockDim.x + threadIdx.x] = row_sum;
  }
}







