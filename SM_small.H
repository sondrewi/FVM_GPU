#ifndef SPARSEMAT_H
#define SPARSEMAT_H

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Sparse Matrix class storing matrix either in
// compressed sparse column (csc) or compressed sparse row format (csr)
class SparseMat {
 public:
  std::string format;

  // Matrix parameters
  int nnz, n_cols, n_rows;

  // Entries and index arrays for matrix
  double* entries = nullptr;
  double* diag = nullptr;
  int* row_idx = nullptr;
  int* col_idx = nullptr;

  // Size of entries array
  int entries_size;

  bool sym = false;
  // Height (nr of rows) of slice
  int slice_height;

  // Number of flices
  int num_slices;

  // Maximal number of entries in a row
  int max_entries_per_row;

  // Arrays giving permutation of
  // rows and columns, relative to original
  int* new_order = nullptr;
  int* new_order_inv = nullptr;
  int* sellp_slice_ptr = nullptr;
  int* slice_mnz = nullptr;
  int* step_lengths = nullptr;
  int* step_starts = nullptr;
  int* step_blocks = nullptr;
  int* nr_rows_by_length = nullptr;
  int* row_starts = nullptr;
  int max_slice_mnz = 0;

  int inner = 0;

  int band = 0;
  int nr_steps;

  // Store old arrays in CSR format when we
  // are interested testing different input
  // parameters without re-reading matrix
  double* old_entries = nullptr;
  int* old_col_idx = nullptr;

  // RHS vector read from file
  double* b = nullptr;

  // Constructor from file
  SparseMat(const std::string& file, const bool read_b, const bool sym_triang);

  // Read from mtx file
  void read_mtx(const std::string& file, const bool read_b,
                const bool sym_triang);

  // Find Cuthill-McKee ordering
  void get_CM();

  bool diag_dom();

  // Convert CSR format to sliced Ellpack format
  void csr_to_sellp(int slice_height_);

  // Order rows according to length
  void order_by_length();

  // Convert CSR format to Staicase format
  void csr_to_staircase(int slice_height_, double alpha);

  // Sort column indices of row under new ordering
  void sort_row(int row, std::vector<int>& col_vec,
                std::vector<double>& entries_vec);

  void csr_to_sellp_orig(int slice_height_);

  void sort_row_orig(int row, std::vector<int>& col_vec,
                     std::vector<double>& entries_vec);

  // Destructor
  ~SparseMat();
};

#endif
