# sort_indices
The functions provided in the main script, `sort_indices.py`, can be used to replace the use of the scipy.sparse matrix method `sort_indices`.

When performing matrix multiplication on scipy.sparse CSR or CSC matrices, the "indices" field of the resulting matrix will generally be unsorted. Such matrix multiplication could be performed, e.g., with SciPy's `@` operator, or the `dot_product_mkl` function from `sparse_dot_mkl` (see https://pypi.org/project/sparse-dot-mkl/). Calling `scipy.sparse.csx_array.sort_indices` is often inefficient. Partly, this has to do with the fact that `sort_indices` will only perform single-thread operations. It's worth noting that the function `dot_product_mkl`, used for the testing in `sort_indices.py`, provides a flag to return a matrix with sorted indices out-of-the-box.

There are two functions defined in this project's main Python file, `sort_indices.py`, that offer some speed-up. The first, `numpy_sort_indices`, sorts the indices as well as the data arrays of a given CSR or CSC matrix using only numpy routines and a for-loop. The second, `numba_sort_indices`, does the same but takes advantage of JIT-compilation and multi-threading provided by the Numba package.

The timings for a 10,000-by-10,000 matrix on my personal machine are about 3.7s for the SciPy-inbuilt `sort_indices`, about 2.7s for `numpy_sort_indices`, and 0.8s for `numba_sort_indices`. The matrix multiplication with `dot_product_mkl` took 0.9s longer when specifying `reorder_output=True`, making `numba_sort_indices` the overall fastest option. In my experience, `numba_sort_indices` performs considerably better the more cores you have available.


# Usage
Simply run the file `sort_indices.py` as a whole to test the included functions and obtain timings for your machine. Alternatively, import functions `numba_sort_indices` or `numpy_sort_indices` individually. For example:
```
from scipy.sparse import rand
from sparse_dot_mkl import dot_product_mkl
from sort_indices import numba_sort_indices


n = int(1e4)
A = rand(n, n, density=0.01, format="csr")
B = rand(n, n, density=0.01, format="csr")

C = dot_product_mkl(A, B)    # creates matrix with unsorted indices

numba_sort_indices(C.indptr, C.indices, C.data)

print(C.has_sorted_indices)    # check that indices are now sorted
```
Using `numpy_sort_indices` works identically.
Make sure not to call `scipy.sparse.csx_array.has_sorted_indices` before actually sorting the indices as this would set the field to `False` and you'd have to manually set it to `True` after sorting indices.


# Dependencies
The packages required are:
* NumPy
* SciPy
* Numba (for Numba-powered sorting function)
* sparse_dot_mkl (for testing sorting functions only, offers multi-threading for matrix products)
