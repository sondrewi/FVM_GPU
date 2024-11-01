#include <cuda_runtime.h>
#include <cusparse.h>
#include "CSLS_small.cuh"
#include <chrono>
#include <fstream>
#include <iostream>
#include <cstdlib>

void mult_CPU(double* Ax, double* x, SparseMat& matA){
  int n_rows = matA.n_rows;

  for (int i = 0; i < n_rows; i++) {
    int row_start = matA.row_idx[i];
    int row_end = matA.row_idx[i + 1];

    double sum = 0;

    for (int j = row_start; j < row_end; j++) {
        sum += matA.entries[j] * x[matA.col_idx[j]];
    }

    Ax[i] = sum;
  }
}

double dot_prod_CPU(double* x, double* y, int n_rows){
    double xTy = 0;

    for(int i=0; i<n_rows; i++){
        xTy += x[i]*y[i];
    }

    return xTy;
}

void add_vecs(double* x, double* y, double* result, double scale, int n_rows){
    for(int i = 0; i < n_rows; i++){
        result[i] = x[i] + scale*y[i];
    }
}

double calc_res_CPU(double* Ax, double* x, double* b, SparseMat& mat){
    double res = 0;
    mult_CPU(Ax, x, mat);

    for(int i = 0; i < mat.n_rows; i++){
        res += (Ax[i]- b[i])*(Ax[i]- b[i]);
    }

    return res;
}

//Solve System of equations Mz = r where M = L L^T but is stored as L + L^T
void solve_triangular(double* M_entries, double* r, double* z, SparseMat& mat){
    int n_rows = mat.n_rows;

    //First perform forward-substitution
    for(int i = 0; i < n_rows; i++){
        z[i] = r[i];
        double M_diag;

        /*if(i < 50){
            std::cout << "z["<< i << "] start value = " << z[i] << std::endl;
        }*/

        for(int k = mat.row_idx[i]; k < mat.row_idx[i+1]; k++){

            if(mat.col_idx[k] < i){
                z[i] -= M_entries[k]*z[mat.col_idx[k]];

                /*if(i < 50){
                    std::cout << "nbr z[" << mat.col_idx[k] << "] = " << z[mat.col_idx[k]] << std::endl;
                    std::cout << "nbr M[" << i << "," << mat.col_idx[k] << "] = " << M_entries[k] << std::endl;
                    std::cout << "nbr zM = " << M_entries[k]*z[mat.col_idx[k]] << std::endl;
                }*/

            }

            else if(mat.col_idx[k] == i){
                M_diag = M_entries[k];
            }
        }

        z[i] /= M_diag;
        /*if(i < 50){
                    std::cout << "M[" << i<<","<<i<<"] = " << M_diag << std::endl;
                    std::cout << "z[" << i << "] = " << z[i] << std::endl;
                    std::cout << std::endl;
        }*/

        /*if(abs(z[i]) > 1e9){
            std::cerr << i << std::endl;
            std::exit(1);  // Return a non-zero status code to indicate an error
        }*/
    }

    //Perform back-substitution
    for(int i = n_rows - 1; i >= 0; i--){
        double M_diag;
        for(int k = mat.row_idx[i+1] - 1; k >= mat.row_idx[i]; k--){
            if(mat.col_idx[k] > i){
                z[i] -= M_entries[k]*z[mat.col_idx[k]];
            }
            else if(mat.col_idx[k] == i){
                M_diag = M_entries[k];
            }
        }

        z[i] /= M_diag;
    }
}

double CG_step_CPU(double* res, double* d, double* Ad, double* x, double rTr, SparseMat& matA, int n_rows){
    mult_CPU(Ad, d, matA);

    double dAd = dot_prod_CPU(d, Ad, n_rows);

    add_vecs(res, Ad, res, -rTr/dAd, n_rows);

    add_vecs(x, d, x, rTr/dAd, n_rows);

    double nrTnr = dot_prod_CPU(res, res, n_rows);
    add_vecs(res, d, d, nrTnr/rTr, n_rows);

    return nrTnr;
}



double PCG_step_CPU(double* res, double* z, double* d, double* Ad, double* x, double* M_entries, double zTr, SparseMat& matA, int n_rows, double& triang_solve_time){
    mult_CPU(Ad, d, matA);
    double dAd = dot_prod_CPU(d, Ad, n_rows);
    add_vecs(res, Ad, res, -zTr/dAd, n_rows);
    add_vecs(x, d, x, zTr/dAd, n_rows);

    auto start_CPU = std::chrono::high_resolution_clock::now();
    solve_triangular(M_entries, res, z, matA);
    auto end_CPU = std::chrono::high_resolution_clock::now();
    triang_solve_time += std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();

    double nzTnr = dot_prod_CPU(z, res, n_rows);
    add_vecs(z, d, d, nzTnr/zTr, n_rows);
    return nzTnr;
}

void GS_sweep_CPU(double* x, double* b, SparseMat& mat, bool fwd_and_bck = false){
  int n_rows = mat.n_rows;

  for (int i = 0; i < n_rows; i++) {
    int z = i;
    if(fwd_and_bck){
        z = n_rows - 1 - i;
    }

    int row_start = mat.row_idx[z];
    int row_end = mat.row_idx[z + 1];

    double sum = 0;

    double diag_val;

    for (int j = row_start; j < row_end; j++) {
      if(mat.col_idx[j] == z){
        diag_val = mat.entries[j];
      }

      else{
        sum += mat.entries[j] * x[mat.col_idx[j]];
      }
    }

    x[z] = (b[z] - sum)/diag_val;
  }
}

void Ichol_CPU(SparseMat& mat, double* ichol_entries, int n_rows){
    for(int i = 0; i < n_rows; i++){
        double A_ii;
        double row_sq_norm = 0;
        int diag_index;

        //Iterate over lower elements in row i
        for(int k = mat.row_idx[i]; k < mat.row_idx[i+1]; k++){
            if(mat.col_idx[k] < i){
                row_sq_norm += pow(ichol_entries[k],2);
            }

            else if(mat.col_idx[k] == i){
                A_ii = mat.entries[k];
                diag_index = k;
                break;
            }
        }

        //Calculated the diagonal elements
        double L_ii = sqrt(A_ii - row_sq_norm);
        ichol_entries[diag_index] = L_ii;

        //Calculate non-zero elements on COLUMN i
        //As A symmetric, we know corresponding rows
        //are the same as non-zero columns above diagonal in row i
        for(int m = mat.row_idx[i]; m < mat.row_idx[i+1]; m++){
            if(mat.col_idx[m] > i){
                //Get the row
                int j = mat.col_idx[m];

                //Get the corresponding element from A
                double A_ji = mat.entries[m];
                double truncated_dot = 0;
                int L_ji_index;

                //Iterate over the row j up until column i
                for(int k = mat.row_idx[j]; k < mat.row_idx[j + 1]; k++){
                    if(mat.col_idx[k] < i){
                        //check for match in column index with entries in row i
                        for(int a = mat.row_idx[i]; a < mat.row_idx[i+1]; a++){
                            if(mat.col_idx[k] == mat.col_idx[a]){
                                truncated_dot += ichol_entries[k]*ichol_entries[a];
                                break;
                            }
                        }
                    }

                    if(mat.col_idx[k] == i){
                        L_ji_index = k;
                    }
                }

                //Add ichol entry to L and L^T (stored in same matrix)
                double L_ji = (A_ji - truncated_dot)/L_ii;
                ichol_entries[m] = L_ji;
                ichol_entries[L_ji_index] = L_ji;
            }
        }
    }
}

void CPU_ichol_PCG(SparseMat& mat, double* res_arr, double* time_arr, double* b, int num_iters, int freq){
    int n_rows = mat.n_rows;

    double* z = new double[n_rows];
    double* d = new double[n_rows];
    double* res = new double[n_rows];
    double* Ad = new double[n_rows];
    double* x = new double[n_rows];
    double* M = new double[mat.nnz];

    for(int i =0; i < n_rows; i++){
        res[i] = b[i];
        res_arr[0] += b[i]*b[i];
    }

    auto start_CPU = std::chrono::high_resolution_clock::now();
    Ichol_CPU(mat, M, n_rows);
    auto end_CPU = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();
    std::cout << "Ichol Factorisation time: " << duration << std::endl;

    start_CPU = std::chrono::high_resolution_clock::now();
    solve_triangular(M, res, z, mat);
    end_CPU = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();

    std::cout << "Triangular Solve Time Single: " << duration << std::endl;

    double res_calc_time = 0;
    double triang_solve_time = 0;

    double zTr = 0;

    for(int i = 0; i < n_rows; i++){
        d[i] = z[i];
        zTr += z[i]*res[i];
    }

    //start_CPU = std::chrono::high_resolution_clock::now();
    for(int k = 0; k < num_iters; k++){
        zTr = PCG_step_CPU(res, z, d, Ad, x, M, zTr, mat, n_rows, triang_solve_time);

        if((k+1)%freq == 0){
            start_CPU = std::chrono::high_resolution_clock::now();
            res_arr[(k+1)/freq] = dot_prod_CPU(res, res, n_rows);
            end_CPU = std::chrono::high_resolution_clock::now();
            //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();
            res_calc_time = std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();
            //time_arr[(k+1)/freq] = time_arr[k/freq] + duration;
            start_CPU = std::chrono::high_resolution_clock::now();
        }
    }

    std::cout << "res_calc_time PCG CPU: " << res_calc_time << std::endl;
    std::cout << "Triangular Solve Time 100: " << triang_solve_time << std::endl;

    delete[] z;
    delete[] d;
    delete[] Ad;
    delete[] x;
    delete[] M;
    delete[] res;
}

void CPU_CG(SparseMat& mat, double* res_arr, double* time_arr, double* b, int num_iters, int freq){
    int n_rows = mat.n_rows;

    double* d = new double[n_rows];
    double* Ad = new double[n_rows];
    double* x = new double[n_rows]();
    double* res = new double[n_rows];

    double rTr = 0;

    for(int i = 0; i < n_rows; i++){
        res[i] = b[i];
        d[i] = b[i];
        rTr += res[i]*res[i];
    }

    res_arr[0] = rTr;

    auto start_CPU = std::chrono::high_resolution_clock::now();
    for(int k = 0; k < num_iters; k++){
        rTr = CG_step_CPU(res, d, Ad, x, rTr, mat, n_rows);

        if((k+1)%freq == 0){
            res_arr[(k+1)/freq] = rTr;
            auto end_CPU = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();
            time_arr[(k+1)/freq] = time_arr[((k+1)/freq) - 1] + duration;
            start_CPU = std::chrono::high_resolution_clock::now();
        }
    }

    delete[] d;
    delete[] Ad;
    delete[] x;
    delete[] res;
}

void GS_CPU(SparseMat& mat, double* res_arr, double* time_arr, double* b, int num_iters, int freq, bool fwd_and_bck){
    int n_rows = mat.n_rows;

    double* x = new double[n_rows]();
    double* Ax = new double[n_rows]();

    res_arr[0] = dot_prod_CPU(b,b, n_rows);

    auto start_CPU = std::chrono::high_resolution_clock::now();
    for(int k = 0; k < num_iters; k++){
        GS_sweep_CPU(x, b, mat, (fwd_and_bck && (k+1)%2 == 0));

        if((k+1)%freq == 0){
            mult_CPU(Ax, x, mat);
            double res_norm = 0;
            for(int i = 0; i < n_rows; i++){
                res_norm += (Ax[i] - b[i])*(Ax[i] - b[i]);
            }

            res_arr[(k+1)/freq] = res_norm;
            auto end_CPU = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();
            time_arr[(k+1)/freq] = time_arr[((k+1)/freq) - 1] + duration;
            start_CPU = std::chrono::high_resolution_clock::now();
        }
    }

    delete[] Ax;
    delete[] x;
}

void CG_GPU(CUDA_SLS& sls, double* res_arr, double* time_arr, int num_iters, int freq){
    sls.prepare_CG();

    double* res_sq = new double[1];
    cudaMemcpy(res_sq, sls.rTr_dev, sizeof(double), cudaMemcpyDeviceToHost);
    res_arr[0] = res_sq[0];

    double res_calc_time = 0;

    //auto start_CPU = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < num_iters; i++){
        sls.CG_step();

        if((i+1)%freq == 0){
            if(sls.is_precond){
                auto start_CPU = std::chrono::high_resolution_clock::now();
                res_arr[(i+1)/freq] = sls.get_orig_res_sq();
                cudaDeviceSynchronize();
                auto end_CPU = std::chrono::high_resolution_clock::now();
                //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();
                res_calc_time += std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();
                //time_arr[(i+1)/freq] = time_arr[i/freq] + duration;
                //start_CPU = std::chrono::high_resolution_clock::now();
            }

            else{
                auto start_CPU = std::chrono::high_resolution_clock::now();
                cudaMemcpy(res_sq, sls.rTr_dev, sizeof(double), cudaMemcpyDeviceToHost);
                res_arr[(i+1)/freq] = res_sq[0];
                cudaDeviceSynchronize();
                auto end_CPU = std::chrono::high_resolution_clock::now();
                //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();
                res_calc_time += std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();
                //time_arr[(i+1)/freq] = time_arr[i/freq] + duration;
                //start_CPU = std::chrono::high_resolution_clock::now();
            }
        }
    }

    std::cout << "res_calc/copy_time GPU: " << res_calc_time << std::endl;

    delete[] res_sq;
}

void smooth_GPU(CUDA_SLS& sls, std::string smoother, double* res_arr, double* time_arr, int num_iters, int freq, bool async, int num_sweeps){
    sls.prepare_smoother(async);
    sls.reset_smoother();

    double* res_sq = new double[1];
    cudaMemcpy(res_sq, sls.rTr_dev, sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_CHECK
    res_arr[0] = res_sq[0];

    auto start_CPU = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < num_iters; i++){
        if(smoother == "Jacobi"){
            sls.smooth_sweeps(1, num_sweeps, 1e-15, async);
        }

        else if(smoother == "CG"){
            sls.smooth_sweeps(0, num_sweeps, 1e-15, async);
        }

        else if(smoother == "GS"){
            sls.smooth_sweeps(2, num_sweeps, 1e-15, async);
        }

        if((i+1)%freq == 0){
            //cudaDeviceSynchronize();
            res_arr[(i+1)/freq] = sls.calc_res();
            cudaDeviceSynchronize();
            auto end_CPU = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();
            time_arr[(i+1)/freq] = time_arr[i/freq] + duration;
            start_CPU = std::chrono::high_resolution_clock::now();
        }
    }

    delete[] res_sq;
}

void time_mult_GPU(std::string mtx, bool sym, std::string algo){

    //First pass with modified format
    CUDA_SLS sls(mtx, false, sym);
    SparseMat& mat = sls.mat;
    int n_rows = mat.n_rows;

    mat.get_CM();

    if(algo == "ellpack_row_based"){
        mat.csr_to_sellp_orig(1024);
    }

    else if(algo == "ellpack_shared"){
        mat.csr_to_sellp(512);
        std::cout << "inner ratio: " << ((double)mat.inner)/(mat.nnz - n_rows)<< std::endl;
    }

    double* b_GPU = new double[n_rows];
    std::fill(b_GPU, b_GPU + n_rows, 1);

    sls.copy_system(algo, b_GPU);
    cudaMalloc((void**)&sls.Ad_dev, n_rows*sizeof(double));
    cudaMalloc((void**)&sls.rTr_dev, sizeof(double));

    double avg_time_mod = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double rTr;

    for(int i = 0; i < 100; i++){
        cudaEventRecord(start);
        sls.mult(sls.Ad_dev, sls.b_dev);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        sls.dot_prod(sls.rTr_dev, sls.Ad_dev);

        if(i == 10){
            double* res = new double[1];
            cudaMemcpy(res, sls.rTr_dev, sizeof(double), cudaMemcpyDeviceToHost);
            rTr = res[0];
            delete[] res;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        avg_time_mod += milliseconds;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Matrix: " << mtx << std::endl;
    std::cout << "rTr: " << rTr << std::endl;
    std::cout << "SELL Avg Time in microseconds: " << avg_time_mod*10 << std::endl;
    std::cout << std::endl;
}

void read_b(std::string b_file, double* b){
  std::ifstream file_b(b_file);

  if (!file_b.is_open()) {
    std::cerr << "Failed to open the mtx file." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Ignore headers and comments:
  while (file_b.peek() == '%') {
    file_b.ignore(2048, '\n');
  }

  int n_rows;
  int cols;

  file_b >> n_rows >> cols;

  for(int i = 0; i < n_rows; i++){
    file_b >> b[i];
  }
}


int main(void) {
    CUDA_SLS sls("100K.mtx", true, false);

    SparseMat& mat = sls.mat;
    int n_rows = mat.n_rows;

    std::cout << "Matrix is diag dom: " << mat.diag_dom() << std::endl;

    double* b_CPU = new double[n_rows];
    for(int i = 0; i < n_rows; i++){
        //b_CPU[i] = (i<100) ? 1 : 0;
        b_CPU[i] = mat.b[i];
    }

    //read_b("parabolic_fem_b.mtx", b_CPU);

    int num_iters = 10000;
    int check_frequency = 100;
    //int num_sweeps = 30;

    int res_length = (num_iters + check_frequency - 1)/check_frequency;

    double* CG_res_CPU = new double[res_length]();
    double* CG_time_CPU = new double[res_length]();
    double* PCG_res_CPU = new double[res_length]();
    double* PCG_time_CPU = new double[res_length]();

    CPU_ichol_PCG(mat, PCG_res_CPU, PCG_time_CPU, b_CPU, num_iters, check_frequency);
    CPU_CG(mat, CG_res_CPU, CG_time_CPU, b_CPU, num_iters, check_frequency);

    std::cout << "CPU CG done" << std::endl;

    std::ofstream cg_cpu("PCG_CPU_G3.dat");
    for(int i = 0; i < res_length; i++){
        cg_cpu << CG_res_CPU[i] << "  " << CG_time_CPU[i] << "  " << PCG_res_CPU[i] << "  " << PCG_time_CPU[i] << std::endl;
    }

    delete[] CG_res_CPU;
    delete[] CG_time_CPU;
    delete[] PCG_res_CPU;
    delete[] PCG_time_CPU;

    mat.get_CM();
    mat.csr_to_sellp(512);

    double* b_GPU = new double[n_rows];
    for(int i = 0; i < n_rows; i++){
        b_GPU[mat.new_order_inv[i]] = b_CPU[i];
    }

    std::string algo = "ellpack_shared";
    sls.copy_system(algo, b_GPU);

    std::cout << "System Copied" << std::endl;

    double* res_CG_GPU = new double[res_length]();
    double* res_RuizCG_GPU = new double[res_length]();
    double* res_DiagCG_GPU = new double[res_length]();

    double* time_CG_GPU = new double[res_length]();
    double* time_RuizCG_GPU = new double[res_length]();
    double* time_DiagCG_GPU = new double[res_length]();

    CG_GPU(sls, res_CG_GPU, time_CG_GPU, num_iters, check_frequency);
    std::cout << sls.is_precond << std::endl;

    std::cout << "normal cg finished" << std::endl;

    sls.prepare_Ruiz_precond();

    auto start_CPU = std::chrono::high_resolution_clock::now();
    int num_ruiz_iter = sls.precond_Ruiz();
    cudaDeviceSynchronize();
    auto end_CPU = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();
    std::cout << "Ruiz Calculation Time: " << duration << std::endl;
    std::cout << "Ruiz Iterations: " << num_ruiz_iter << std::endl;

    std::cout << "Ruiz Finished" << std::endl;

    CG_GPU(sls, res_RuizCG_GPU, time_RuizCG_GPU, num_iters, check_frequency);

    sls.undo_precond();
    sls.prepare_diag_precond();

    start_CPU = std::chrono::high_resolution_clock::now();
    sls.diag_precond();
    cudaDeviceSynchronize();
    end_CPU = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_CPU - start_CPU).count();
    std::cout << "Diag Preconditioner Calculation Time: " << duration << std::endl;

    CG_GPU(sls, res_DiagCG_GPU, time_DiagCG_GPU, num_iters, check_frequency);

    // Output file
    std::ofstream cg_GPU("PCG_GPU_G3.dat");

    for(int i = 0; i < res_length; i++){
         cg_GPU << res_CG_GPU[i] << "  " << time_CG_GPU[i] << "  " << res_RuizCG_GPU[i] << "  "
         << time_RuizCG_GPU[i] << "  " << res_DiagCG_GPU[i] << "  " << time_DiagCG_GPU[i] << std::endl;
    }

    /*double* CG_res_CPU = new double[res_length]();
    double* CG_res_GPU = new double[res_length]();
    double* CG_prec_res_CPU = new double[res_length]();
    double* CG_prec_res_GPU = new double[res_length]();

    double* res_GS_CPU = new double[res_length]();
    double* res_GS_CPU_fb = new double[res_length]();
    double* res_GS_GPU = new double[res_length]();
    double* res_GS_GPU_async = new double[res_length]();
    double* res_Jacobi_GPU = new double[res_length]();
    double* res_Jacobi_GPU_async = new double[res_length]();
    double* res_local_CG_GPU = new double[res_length]();
    double* res_local_CG_GPU_async = new double[res_length]();

    double* time_GS_CPU = new double[res_length]();
    double* time_GS_CPU_fb = new double[res_length]();
    double* time_GS_GPU = new double[res_length]();
    double* time_GS_GPU_async = new double[res_length]();
    double* time_Jacobi_GPU = new double[res_length]();
    double* time_Jacobi_GPU_async = new double[res_length]();
    double* time_local_CG_GPU = new double[res_length]();
    double* time_local_CG_GPU_async = new double[res_length]();

    GS_CPU(mat, res_GS_CPU, time_GS_CPU, b_CPU, num_iters, check_frequency, false);
    GS_CPU(mat, res_GS_CPU_fb, time_GS_CPU_fb, b_CPU, num_iters, check_frequency, true);

    mat.get_CM();
    mat.csr_to_sellp(512);

    std::cout << "max slice mnz: " << mat.max_slice_mnz << std::endl;

    std::cout << "reordering done" << std::endl;

    double* b_GPU = new double[n_rows];
    for(int i = 0; i < n_rows; i++){
        b_GPU[mat.new_order_inv[i]] = b_CPU[i];
    }

    std::string algo = "ellpack_shared";
    sls.copy_system(algo, b_GPU);

    std::cout << "System copied" << std::endl;

    //CG_GPU(sls, CG_res_GPU, 20000, check_frequency);


    smooth_GPU(sls, "Jacobi", res_Jacobi_GPU, time_Jacobi_GPU, num_iters, check_frequency, false, 100);
    std::cout << "Jacobi sync done" << std::endl;
    smooth_GPU(sls, "Jacobi", res_Jacobi_GPU_async, time_Jacobi_GPU_async, num_iters, check_frequency, true, 100);
    std::cout << "Jacobi async done" << std::endl;
    smooth_GPU(sls, "GS", res_GS_GPU, time_GS_GPU, num_iters, check_frequency, false, 100);
    std::cout << "GS sync done" << std::endl;
    smooth_GPU(sls, "GS", res_GS_GPU_async, time_GS_GPU_async, num_iters, check_frequency, true, 100);
    std::cout << "GS async done" << std::endl;
    smooth_GPU(sls, "CG", res_local_CG_GPU, time_local_CG_GPU, num_iters, check_frequency, false, 100);
    std::cout << "CG sync done" << std::endl;
    smooth_GPU(sls, "CG", res_local_CG_GPU_async, time_local_CG_GPU_async, num_iters, check_frequency, true, 100);
    std::cout << "CG async done" << std::endl;

    // Output file
    std::ofstream smooth_cpu("Smooth_CPU_pfem.dat");
    std::ofstream smooth_GPU("Smooth_GPU_pfem.dat");

    //sls.diag_precond();
    //CG_GPU(sls, CG_prec_res_GPU, num_iters, check_frequency);

    for(int i = 0; i < res_length; i++){
         smooth_cpu << res_GS_CPU[i] << "  " << time_GS_CPU[i] << "  " << res_GS_CPU_fb[i] << "  " << time_GS_CPU_fb[i] << std::endl;
         smooth_GPU << res_Jacobi_GPU[i] << "  " << time_Jacobi_GPU[i] << "  " << res_Jacobi_GPU_async[i] << "  "
         << time_Jacobi_GPU_async[i] << "  " << res_GS_GPU[i] << "  " << time_GS_GPU[i] << "  " << res_GS_GPU_async[i] <<
         "  " << time_GS_GPU_async[i] << "  " << res_local_CG_GPU[i] << "  " << time_local_CG_GPU[i] <<
         "  " << res_local_CG_GPU_async[i] << "  " << time_local_CG_GPU_async[i] << std::endl;
    }*/

    /*std::vector<std::string> mats = {"CurlCurl_4.mtx", "G3_circuit.mtx", "ecology2.mtx",
    "100K.mtx", "thermal2.mtx", "tmt_sym.mtx", "apache2.mtx", "parabolic_fem.mtx"};
    bool sym_storage[8] = {true, true, true, false, true, true, true, true};

    for(int i = 0; i < 8; i++){
        time_mult_GPU(mats[i], sym_storage[i], "ellpack_shared");
    }

    time_mult_GPU("thermomech_dM.mtx", true, "ellpack_shared");*/
}
