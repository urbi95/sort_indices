import numpy as np
import numba as nb


def numpy_sort_indices(indptr, indices, data):
    for i in range(indptr.size - 1):
        col_indices = indices[indptr[i]:indptr[i+1]]
        col_data = data[indptr[i]:indptr[i+1]]

        idx = np.argsort(col_indices)
        indices[indptr[i]:indptr[i+1]] = col_indices[idx]
        data[indptr[i]:indptr[i+1]] = col_data[idx]


@nb.njit((nb.i4[:], nb.i4[:], nb.f8[:]), parallel=True)
def numba_sort_indices(indptr, indices, data):
    for i in nb.prange(indptr.size - 1):
        col_indices = indices[indptr[i]:indptr[i+1]]
        col_data = data[indptr[i]:indptr[i+1]]

        idx = np.argsort(col_indices)
        indices[indptr[i]:indptr[i+1]] = col_indices[idx]
        data[indptr[i]:indptr[i+1]] = col_data[idx]


if __name__=="__main__":
    from scipy.sparse import rand
    from time import time
    from sparse_dot_mkl import dot_product_mkl


    def check_sparse_matrices_identical(*args):
        for i in range(len(args) - 1):
            if not check_two_sparse_matrices_identical(args[i], args[i + 1]):
                return False
        
        return True
    

    def check_two_sparse_matrices_identical(A, B):
        try:
            diff = A - B    # check size
        except:
            return False
        if diff.nnz > 0:    # check data
            return False
        if not np.array_equal(A.indptr, B.indptr):
            return False
        if not np.array_equal(A.indices, B.indices):
            return False
        if not A.format == B.format:
            return False
        if not A.dtype == B.dtype:
            return False
        if not A.has_sorted_indices == B.has_sorted_indices:
            return False
        
        return True


    # test numpy_sort_indices and numba_sort_indices in practice
    n = int(1e4)
    A = rand(n, n, density=0.01, format="csr")
    B = rand(n, n, density=0.01, format="csr")

    mkl_t = time()
    C = dot_product_mkl(A, B)    # dot product leads to unsorted indices
    mid = time()
    mkl_C = dot_product_mkl(A, B, reorder_output=True)    # dot_product_mkl provides flag to sort indices
    mkl_t = time() - mid - (mid - mkl_t)    # time difference between reorder_output=True and reorder_output=False

    scipy_C = C.copy()
    scipy_t = time()
    scipy_C.sort_indices()
    scipy_t = time() - scipy_t

    numpy_C = C.copy()
    numpy_t = time()
    numpy_sort_indices(numpy_C.indptr, numpy_C.indices, numpy_C.data)
    numpy_t = time() - numpy_t

    numba_C = C.copy()
    numba_t = time()
    numba_sort_indices(numba_C.indptr, numba_C.indices, numba_C.data)
    numba_t = time() - numba_t

    success = check_sparse_matrices_identical(scipy_C, numpy_C, numba_C, mkl_C)
    if success:
        print("Success. All resulting matrices were identical!\n")
    else:
        print("FAILURE. Resulting matrices were not all identical.\n")

    print( "          |   SciPy   |   NumPy   |   Numba   |    MKL")
    print(f"Run times |{scipy_t: 8.3f}s  |{numpy_t: 8.3f}s  |{numba_t: 8.3f}s  |{mkl_t: 8.3f}s")
