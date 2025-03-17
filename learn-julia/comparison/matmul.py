from random import random
import time


def printvec(vec):
    print("[", end="")
    for i in range(len(vec) - 1):
        print(f"{vec[i]:.6f}, ", end="")
    print(f"{vec[len(vec)-1]:.6f}", end="")
    print("]")


def printmat(mat):
    for i in range(len(mat)):
        printvec(mat[i])


def randmat(m, n):
    return [[random() for _ in range(n)] for _ in range(m)]


def zeromat(m, n):
    return [[0 for _ in range(n)] for _ in range(m)]


def matmul(A, B):
    n_rows_A = len(A)
    n_cols_A = len(A[0])

    n_rows_B = len(B)
    n_cols_B = len(B[0])

    if n_cols_A != n_rows_B:
        raise RuntimeError("Cannot multiply mismatched matrices!")

    n_rows_C = n_rows_A
    n_cols_C = n_cols_B
    C = zeromat(n_rows_C, n_cols_C)
    for i in range(n_rows_C):
        for j in range(n_cols_C):
            C[i][j] = 0
            for k in range(n_cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C


def main():
    m, n, p = 500, 500, 500
    with open("py_myimpl.txt", "wt") as f:
        for _ in range(100):
            A = randmat(m, n)
            B = randmat(n, p)

            start = time.process_time()
            C = matmul(A, B)  # noqa
            end = time.process_time()

            proc_time = end - start
            print(proc_time, file=f)

    print("Done")


if __name__ == "__main__":
    main()
