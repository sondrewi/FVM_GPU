#include "SM_small.H"

// Constructor for read from file
SparseMat::SparseMat(const std::string& file, const bool read_b,
                     const bool sym_triang) {
  read_mtx(file, read_b, sym_triang);
}

bool SparseMat::diag_dom() {
  bool diag_dom = true;
  bool one_strict = false;

  for (int i = 0; i < n_rows; i++) {
    int row_start = row_idx[i];
    int row_end = row_idx[i + 1];

    double off_diag_sum = 0;
    double diag;

    for (int j = row_start; j < row_end; j++) {
      if (col_idx[j] == i) {
        diag = entries[j];
      }

      else {
        off_diag_sum += abs(entries[j]);
      }
    }

    if (off_diag_sum > abs(diag)) {
      diag_dom = false;
      break;
    }

    else if (off_diag_sum < abs(diag) && !one_strict) {
      one_strict = true;
    }
  }

  return (diag_dom && one_strict);
}

// Read from mtx file into csr format
void SparseMat::read_mtx(const std::string& file, const bool read_b,
                         const bool sym_triang) {
  // read_b: RHS of sparse linear system is contained in bottom of .mtx file
  // sym_triang: only diagonal and upper (or lower) coefficients stored in file

  if (sym_triang || file == "100K.mtx") {
    sym = true;
  }

  std::ifstream mtx(file);

  if (!mtx.is_open()) {
    std::cerr << "Failed to open the mtx file." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Default format is taken as csr
  format = "csr";

  // Ignore headers and comments:
  while (mtx.peek() == '%') {
    mtx.ignore(2048, '\n');
  }

  // Read number of rows, columns and non-zeros:
  mtx >> n_rows >> n_cols >> nnz;

  int row;
  int col;
  double val;

  // Vectors to hold entries and indices
  std::vector<std::vector<double> > entries_vec(n_rows);
  std::vector<int> row_idx_vec(n_rows, 0);
  std::vector<std::vector<int> > col_idx_vec(n_rows);

  for (int l = 0; l < nnz; l++) {
    mtx >> row >> col >> val;

    // File is in one-base index
    // Add to number of entries in row
    row_idx_vec[row - 1] += 1;

    bool set_col = false;

    // Push back in appropriate col_idx_vec and entries_vec
    col_idx_vec[row - 1].push_back(col - 1);
    entries_vec[row - 1].push_back(val);

    // If matrix is symmetric and only upper (or lower) entries provided
    // insert mirrored entry
    if (sym_triang && col != row) {
      // Add to number of entries in mirrored row
      row_idx_vec[col - 1] += 1;

      // Push back in appropriate col_idx_vec and entries_vec
      col_idx_vec[col - 1].push_back(row - 1);
      entries_vec[col - 1].push_back(val);
    }
  }

  // Must change nnz if symmetric
  if (sym_triang) {
    nnz += nnz - n_rows;
  }

  // Initiate arrays for csr storage
  entries = new double[nnz];
  col_idx = new int[nnz];
  row_idx = new int[n_rows + 1]();

  max_entries_per_row = 0;

  // Iterate over vectors and read to arrays
  for (int i = 0; i < n_rows; i++) {
    row_idx[i + 1] = row_idx[i] + row_idx_vec[i];

    // Keep track of maximal number of non-zeros by row
    if (row_idx_vec[i] > max_entries_per_row) {
      max_entries_per_row = row_idx_vec[i];
    }

    // Insert entries and column index values
    for (int j = 0; j < row_idx_vec[i]; j++) {
      col_idx[row_idx[i] + j] = col_idx_vec[i][j];
      entries[row_idx[i] + j] = entries_vec[i][j];
    }
  }

  // If b contained in matrix file
  if (read_b) {
    b = new double[n_rows];
    // iterate over lines in mtx file
    std::string char_b_i;
    for (int l = 0; l < n_rows; l++) {
      mtx >> char_b_i;

      b[l] = std::stod(char_b_i);
    }
  }
}

// Destructor
SparseMat::~SparseMat() {
  if (entries != nullptr) {
    delete[] entries;
  }

  if (row_idx != nullptr) {
    delete[] row_idx;
  }

  if (col_idx != nullptr) {
    delete[] col_idx;
  }

  if (b != nullptr) {
    delete[] b;
  }

  if (sellp_slice_ptr != nullptr) {
    delete[] sellp_slice_ptr;
  }

  if (step_lengths != nullptr) {
    delete[] step_lengths;
    step_lengths = nullptr;
  }

  if (step_starts != nullptr) {
    delete[] step_starts;
    step_starts = nullptr;
  }

  if (step_blocks != nullptr) {
    delete[] step_blocks;
    step_blocks = nullptr;
  }

  if (nr_rows_by_length != nullptr) {
    delete[] nr_rows_by_length;
    nr_rows_by_length = nullptr;
  }

  if (row_starts != nullptr) {
    delete[] row_starts;
    row_starts = nullptr;
  }

  if (diag != nullptr) {
    delete[] diag;
    diag = nullptr;
  }

  if (new_order != nullptr) {
    delete[] new_order;
    new_order = nullptr;
  }

  if (new_order_inv != nullptr) {
    delete[] new_order_inv;
    new_order_inv = nullptr;
  }
}

// Compute Cuthill-Mckee order of rows
// Order rows into levels relative to distance from
// zero row with nr of non-zeros in row as tie-breaker
// Always assume rows are first ordered by length
void SparseMat::get_CM() {
  if (format != "csr") {
    std::cout << "format: " << format << std::endl;
    std::cerr << "Matrix is not in csr format" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Order rows into levels relative to distance from
  // zero row with nr of non-zeros in row as tie-breaker
  bool* visited = new bool[n_rows];
  std::fill(visited, visited + n_rows, false);

  int ordered_count = 0;
  int iteration_count = 0;

  int* nr_edges = new int[n_rows];

  int* RCM_order = new int[n_rows];
  int* RCM_order_inv = new int[n_rows];

  int least_edges = max_entries_per_row;
  int start_node;

  // Start by placing rows with no or one entry
  // first. Find also the node, with the least nr of edges > 0
  for (int i = 0; i < n_rows; i++) {
    int edge_nr = row_idx[i + 1] - row_idx[i];

    if (edge_nr <= 1) {
      RCM_order[ordered_count] = i;
      RCM_order_inv[i] = ordered_count;
      ordered_count += 1;
      iteration_count += 1;
      visited[i] = true;
      nr_edges[i] = 0;
    }

    else if (edge_nr < least_edges) {
      start_node = i;
      least_edges = row_idx[i + 1] - row_idx[i];
    }
  }

  // Add start node to the order
  RCM_order[ordered_count] = start_node;
  RCM_order_inv[start_node] = ordered_count;
  ordered_count += 1;
  visited[start_node] = true;
  nr_edges[start_node] = row_idx[start_node + 1] - row_idx[start_node];

  // Continue Breadth-First Search until queue is empty
  while (ordered_count < n_rows) {
    int node = RCM_order[iteration_count];

    // Vector to hold neighbours of current node
    std::vector<int> adjacency_ordering;

    // Traverse all adjacent nodes of the current node
    for (int i = row_idx[node]; i < row_idx[node + 1]; i++) {
      int nbr = col_idx[i];

      // If neighbor has not been visited yet
      if (!visited[nbr]) {
        // Mark neighbor as visited
        visited[nbr] = true;

        // Assign nr of edges of neighbour
        nr_edges[nbr] = row_idx[nbr + 1] - row_idx[nbr] - 1;

        bool set = false;

        // Iterate over nodes in this adjacency list (already sorted) and
        // insert nbr when its number of edges is lower than next node
        // already in list or when order lower than all those already in list
        for (int l = 0; l < adjacency_ordering.size(); l++) {
          int other_node = adjacency_ordering[l];

          if (nr_edges[nbr] < nr_edges[other_node]) {
            adjacency_ordering.insert(adjacency_ordering.begin() + l, nbr);
            set = true;
            break;
          }

          // Sort by original index if nr of edges equal
          else if (nr_edges[nbr] == nr_edges[other_node] && nbr < other_node) {
            adjacency_ordering.insert(adjacency_ordering.begin() + l, nbr);
            set = true;
            break;
          }
        }

        // If node not placed yet
        if (set == false) {
          adjacency_ordering.push_back(nbr);
        }
      }
    }

    // Loop over ordered neighbours in adjacency list
    for (int k = 0; k < adjacency_ordering.size(); k++) {
      int nbr = adjacency_ordering[k];

      RCM_order[ordered_count] = nbr;
      RCM_order_inv[nbr] = ordered_count;

      ordered_count += 1;
    }

    // Increment band if necessary
    if (ordered_count - 1 - iteration_count > band) {
      band = ordered_count - 1 - iteration_count;
    }

    iteration_count += 1;
  }

  std::swap(RCM_order, new_order);
  std::swap(RCM_order_inv, new_order_inv);

  delete[] RCM_order;
  delete[] RCM_order_inv;

  delete[] visited;
  delete[] nr_edges;
}

// Order rows by length
// If RCM order already computed, rows ordered within each length according to
// RCM
void SparseMat::order_by_length() {
  int* length_order = new int[n_rows];
  int* length_order_inv = new int[n_rows];

  std::vector<std::vector<int> > idx_vec(max_entries_per_row + 1);
  nr_rows_by_length = new int[max_entries_per_row + 1]();

  for (int i = 0; i < n_rows; i++) {
    int length = row_idx[new_order[i] + 1] - row_idx[new_order[i]];
    idx_vec[length].push_back(new_order[i]);
    nr_rows_by_length[length] += 1;
  }

  int count = 0;
  for (int i = 0; i < max_entries_per_row + 1; i++) {
    for (int j = 0; j < idx_vec[i].size(); j++) {
      length_order[count] = idx_vec[i][j];
      length_order_inv[idx_vec[i][j]] = count;
      count += 1;
    }
  }

  std::swap(length_order, new_order);
  std::swap(length_order_inv, new_order_inv);

  delete[] length_order;
  delete[] length_order_inv;
}

// Store in the staircase format computed in funcion above
void SparseMat::csr_to_staircase(int slice_height_, double alpha) {
  step_blocks = new int[max_entries_per_row + 1]();

  slice_height = slice_height_;

  // Calculate new number of rows per group/step
  int step_count = 0;
  for (int i = 0; i < max_entries_per_row + 1; i++) {
    if (i == max_entries_per_row) {
      int n = nr_rows_by_length[i];
      step_blocks[step_count] = (n + slice_height - 1) / slice_height;
      nr_rows_by_length[i] =
          slice_height * ((n + slice_height - 1) / slice_height);
    }

    else if (((double)nr_rows_by_length[i] / n_rows) < alpha) {
      nr_rows_by_length[i + 1] += nr_rows_by_length[i];
      nr_rows_by_length[i] = 0;
    }

    else {
      int n = nr_rows_by_length[i];
      step_blocks[step_count] = n / slice_height;
      nr_rows_by_length[i + 1] += n - (n / slice_height) * slice_height;
      nr_rows_by_length[i] = (n / slice_height) * slice_height;
      step_count += 1;
    }
  }

  // Calculate number groups remaining with positive number of rows
  nr_steps = 0;
  for (int i = 0; i < max_entries_per_row + 1; i++) {
    if (nr_rows_by_length[i] > 0) {
      nr_steps += 1;
    }
  }

  // Push block and nr_rows_by_length left
  step_lengths = new int[nr_steps];
  int* new_nr_rows_by_length = new int[nr_steps];
  int* new_blocks = new int[nr_steps];
  int nr_count = 0;
  for (int i = 0; i < max_entries_per_row + 1; i++) {
    if (nr_rows_by_length[i] > 0) {
      new_blocks[nr_count] = step_blocks[nr_count];
      new_nr_rows_by_length[nr_count] = nr_rows_by_length[i];
      step_lengths[nr_count] = i;
      nr_count += 1;
    }
  }

  std::swap(new_blocks, step_blocks);
  std::swap(new_nr_rows_by_length, nr_rows_by_length);
  delete[] new_blocks;
  delete[] new_nr_rows_by_length;

  entries_size = 0;

  // Arrays for first row in each group and index in entries array
  row_starts = new int[nr_steps + 1]();
  step_starts = new int[nr_steps + 1]();

  for (int i = 0; i < nr_steps; i++) {
    int length = step_lengths[i];
    entries_size += length * nr_rows_by_length[i];
    step_starts[i + 1] = entries_size;
    row_starts[i + 1] = row_starts[i] + nr_rows_by_length[i];
  }

  std::cout << entries_size << std::endl;

  // Arrays to hold entries and column indices in new order
  double* new_entries = new double[entries_size]();
  int* new_col_idx = new int[entries_size];
  std::fill(new_col_idx, new_col_idx + entries_size, -1);

  int row_count = 0;
  int slice_first_row;

  for (int i = 0; i < nr_steps; i++) {
    int length = step_lengths[i];

    for (int j = 0; j < nr_rows_by_length[i]; j++) {
      if (row_count < n_rows) {
        int a = new_order[row_count];  // this is the original index

        if (j % slice_height == 0) {
          slice_first_row = row_count;
        }

        int row_nnz = row_idx[a + 1] - row_idx[a];

        std::vector<int> col_vec;
        std::vector<double> entries_vec;

        sort_row(a, col_vec, entries_vec);

        for (int k = 0; k < col_vec.size(); k++) {
          int slice_idx = j / slice_height;
          int idx = step_starts[i] + (slice_idx * length * slice_height) +
                    (k * slice_height) + j % slice_height;

          new_entries[idx] = entries_vec[k];
          new_col_idx[idx] = col_vec[k];
        }
        row_count += 1;
      }
    }
  }

  std::swap(entries, new_entries);
  std::swap(col_idx, new_col_idx);

  delete[] new_entries;
  delete[] new_col_idx;
  delete[] row_idx;

  row_idx = nullptr;
}

// Function to convert CSR to SELL-P format
void SparseMat::csr_to_sellp(int slice_height_) {
  if (format != "csr") {
    std::cout << "format: " << format << std::endl;
    std::cerr << "Matrix is not in csr format" << std::endl;
    exit(EXIT_FAILURE);
  }

  format = "sellp";

  slice_height = slice_height_;

  // Step 1: Calculate number of slices
  num_slices = (n_rows + slice_height - 1) / slice_height;

  // Step 2: Initialize the slice pointer array
  sellp_slice_ptr = new int[num_slices + 1]();

  // Step 3: Count the non-zeros per row
  int* nnz_per_row = new int[n_rows]();
  for (int i = 0; i < n_rows; i++) {
    nnz_per_row[new_order_inv[i]] = row_idx[i + 1] - row_idx[i] - 1;
  }

  int* slice_lengths = new int[num_slices]();
  for (int i = 0; i < n_rows; i++) {
    int slice = i / slice_height;
    slice_lengths[slice] = std::max(slice_lengths[slice], nnz_per_row[i]);
  }

  // Step 5: Fill the slice pointer array
  for (int slice = 0; slice < num_slices; slice++) {
    sellp_slice_ptr[slice + 1] =
        sellp_slice_ptr[slice] + slice_lengths[slice] * slice_height;
  }

  slice_mnz = new int[num_slices]();
  entries_size = sellp_slice_ptr[num_slices];

  // Step 6: Initialize column index and value arrays
  int* new_col_idx = new int[entries_size]();
  double* new_entries = new double[entries_size]();
  diag = new double[n_rows];

  // Step 7: Fill column index and value arrays
  for (int row = 0; row < n_rows; row++) {
    int slice = new_order_inv[row] / slice_height;
    int slice_first_row = slice * slice_height;
    int row_nnz = 0;

    std::vector<int> col_vec;
    std::vector<double> entries_vec;

    sort_row(row, col_vec, entries_vec);

    diag[new_order_inv[row]] = entries_vec[0];

    for (int j = 1; j < col_vec.size(); j++) {
      int idx = sellp_slice_ptr[slice] + (j - 1) * slice_height +
                new_order_inv[row] % slice_height;

      int relative_col = col_vec[j] - slice_first_row;

      if (relative_col >= 0 && relative_col < slice_height) {
        new_col_idx[idx] = -(col_vec[j] - slice_first_row + 1);
        inner += 1;
        row_nnz += 1;
      }

      else {
        new_col_idx[idx] = col_vec[j] + 1;
      }

      new_entries[idx] = entries_vec[j];
    }

    if (row_nnz > slice_mnz[slice]) {
      slice_mnz[slice] = row_nnz;
    }
  }

  for (int slice = 0; slice < num_slices; slice++) {
    if (slice_mnz[slice] > max_slice_mnz) {
      max_slice_mnz = slice_mnz[slice];
    }
  }

  std::swap(entries, new_entries);
  std::swap(col_idx, new_col_idx);

  delete[] new_entries;
  delete[] new_col_idx;

  delete[] nnz_per_row;
  delete[] slice_lengths;
}

// Sort neighbours on a row by new ordering and return row entries and column
// indices in corresponding vectors
// Diagonal value placed first
void SparseMat::sort_row(int row, std::vector<int>& col_vec,
                         std::vector<double>& entries_vec) {
  double diag_value;

  for (int c = row_idx[row]; c < row_idx[row + 1]; c++) {
    bool set = false;

    if (col_idx[c] == row) {
      diag_value = entries[c];
      continue;
    }

    for (int m = 0; m < col_vec.size(); m++) {
      if (new_order_inv[col_idx[c]] < col_vec[m]) {
        col_vec.insert(col_vec.begin() + m, new_order_inv[col_idx[c]]);
        entries_vec.insert(entries_vec.begin() + m, entries[c]);
        set = true;
        break;
      }
    }

    if (set == false) {
      col_vec.push_back(new_order_inv[col_idx[c]]);
      entries_vec.push_back(entries[c]);
    }
  }

  col_vec.insert(col_vec.begin(), new_order_inv[row]);
  entries_vec.insert(entries_vec.begin(), diag_value);
}

// Function to convert CSR to SELL-P format
void SparseMat::csr_to_sellp_orig(int slice_height_) {
  if (format != "csr") {
    std::cout << "format: " << format << std::endl;
    std::cerr << "Matrix is not in csr format" << std::endl;
    exit(EXIT_FAILURE);
  }

  format = "sellp";

  slice_height = slice_height_;

  // Step 1: Calculate number of slices
  num_slices = (n_rows + slice_height - 1) / slice_height;

  // Step 2: Initialize the slice pointer array
  sellp_slice_ptr = new int[num_slices + 1]();

  // Step 3: Count the non-zeros per row
  int* nnz_per_row = new int[n_rows]();
  for (int i = 0; i < n_rows; i++) {
    nnz_per_row[new_order_inv[i]] = row_idx[i + 1] - row_idx[i];
  }

  int* slice_lengths = new int[num_slices]();
  for (int i = 0; i < n_rows; i++) {
    int slice = i / slice_height;
    slice_lengths[slice] = std::max(slice_lengths[slice], nnz_per_row[i]);
  }

  // Step 5: Fill the slice pointer array
  for (int slice = 0; slice < num_slices; slice++) {
    sellp_slice_ptr[slice + 1] =
        sellp_slice_ptr[slice] + slice_lengths[slice] * slice_height;
  }

  entries_size = sellp_slice_ptr[num_slices];

  // Step 6: Initialize column index and value arrays
  int* new_col_idx = new int[entries_size]();
  std::fill(new_col_idx, new_col_idx + entries_size, -1);
  double* new_entries = new double[entries_size]();

  // Step 7: Fill column index and value arrays
  for (int row = 0; row < n_rows; row++) {
    int slice = new_order_inv[row] / slice_height;

    std::vector<int> col_vec;
    std::vector<double> entries_vec;

    sort_row_orig(row, col_vec, entries_vec);

    for (int j = 0; j < col_vec.size(); j++) {
      int idx = sellp_slice_ptr[slice] + j * slice_height +
                new_order_inv[row] % slice_height;

      new_col_idx[idx] = col_vec[j];
      new_entries[idx] = entries_vec[j];
    }
  }

  std::swap(entries, new_entries);
  std::swap(col_idx, new_col_idx);

  delete[] new_entries;
  delete[] new_col_idx;

  delete[] nnz_per_row;
  delete[] slice_lengths;
}

void SparseMat::sort_row_orig(int row, std::vector<int>& col_vec,
                              std::vector<double>& entries_vec) {
  for (int c = row_idx[row]; c < row_idx[row + 1]; c++) {
    bool set = false;

    for (int m = 0; m < col_vec.size(); m++) {
      if (new_order_inv[col_idx[c]] < col_vec[m]) {
        col_vec.insert(col_vec.begin() + m, new_order_inv[col_idx[c]]);
        entries_vec.insert(entries_vec.begin() + m, entries[c]);
        set = true;
        break;
      }
    }

    if (set == false) {
      col_vec.push_back(new_order_inv[col_idx[c]]);
      entries_vec.push_back(entries[c]);
    }
  }
}
