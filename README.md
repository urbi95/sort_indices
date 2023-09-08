# sort_indices
The functions provided in the main script, `sort_indices.py`, can be used to replace the use of the scipy.sparse matrix method `sort_indices`.

When performing matrix multiplication on scipy.sparse CSR or CSC matrices, the "indices" field of the resulting matrix will generally be unsorted. Calling, e.g., `scipy.sparse.csr_array.sort_indices` is often inefficient. This in parts has to do with the fact that `sort_indices` will only perform single-thread operations.

There are two functions defined in this project's main Python file, `sort_indices.py`, that offer some speed-up. The first, `numpy_sort_indices`, sorts the indices as well as the data arrays of a given CSR or CSC matrix using only numpy routines and a for-loop. The second, `numba_sort_indices`, does the same but takes advantage of JIT-compilation and multi-threading provided by the Numba package.

The timings for a 10,000-by-10,000 matrix on my modest 4-core machine are about 5s for the SciPy-inbuilt `sort_indices`, about 4s for `numpy_sort_indices`, and 2s for `numba_sort_indices`. In my experience, `numba_sort_indices` generally performs better the more cores you have.


# Dependencies
The packages required are:
* NumPy
* SciPy
* Numba (for Numba-powered sorting function)
* sparse_dot_mkl (for testing sorting functions only, offers multi-threading for matrix products)
