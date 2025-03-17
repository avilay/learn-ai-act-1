import time
import numpy as np

rng = np.random.default_rng()


def matmul(a, b):
    return a @ b


def main():
    m, n, p = 500, 500, 500

    with open("py_withnp.txt", "wt") as f:
        for _ in range(100):
            a = rng.random((m, n))
            b = rng.random((n, p))

            start = time.process_time()
            c = matmul(a, b)  # noqa
            end = time.process_time()

            proc_time = end - start
            print(proc_time, file=f)

    print("Done")


if __name__ == "__main__":
    main()
