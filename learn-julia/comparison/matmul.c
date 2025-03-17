#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Matrix {
  float **X;
  int numRows;
  int numCols;
} Matrix;

void printvec(float *x, int n) {
  printf("[");
  for (int i = 0; i < n - 1; i++) {
    printf("%f, ", x[i]);
  }
  // last element in the row should not have a trailing comma
  printf("%f", x[n - 1]);
  printf("]\n");
}

void printmat(Matrix mat) {
  for (int i = 0; i < mat.numRows; i++) {
    printvec(mat.X[i], mat.numCols);
  }
}

Matrix randmat(int m, int n) {
  Matrix mat;
  mat.X = (float **)malloc(sizeof(float *) * m);
  for (int i = 0; i < m; i++) {
    mat.X[i] = (float *)malloc(sizeof(float) * n);
    for (int j = 0; j < n; j++) {
      mat.X[i][j] = (float)random() / RAND_MAX;
    }
  }
  mat.numRows = m;
  mat.numCols = n;
  return mat;
}

Matrix zeromat(int m, int n) {
  Matrix mat;
  mat.X = (float **)malloc(sizeof(float *) * m);
  for (int i = 0; i < m; i++) {
    mat.X[i] = (float *)malloc(sizeof(float) * n);
    for (int j = 0; j < n; j++) {
      mat.X[i][j] = 0.0f;
    }
  }
  mat.numRows = m;
  mat.numCols = n;
  return mat;
}

Matrix matmul(Matrix A, Matrix B) {
  if (A.numCols != B.numRows) {
    printf("Unable to multiply mismatched matrices!");
    exit(-1);
  }

  Matrix C = zeromat(A.numRows, B.numCols);
  for (int i = 0; i < C.numRows; i++) {
    for (int j = 0; j < C.numCols; j++) {
      C.X[i][j] = 0;
      for (int k = 0; k < A.numCols; k++) {
        C.X[i][j] += A.X[i][k] * B.X[k][j];
      }
    }
  }
  return C;
}

void freemat(Matrix M) {
  for (int i = 0; i < M.numRows; i++) {
    free(M.X[i]);
  }
  free(M.X);
}

int main(int argc, char **argv) {
  int m = 500;
  int n = 500;
  int p = 500;

  Matrix A, B, C;
  struct timespec start, end;

  FILE *f = fopen("c_myimpl.txt", "wt");
  if (f == NULL) {
    printf("Unable to open file!");
    exit(-1);
  }

  for (int i = 0; i < 100; i++) {
    A = randmat(m, n);
    B = randmat(n, p);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    C = matmul(A, B);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

    long nanosecs =
        (end.tv_sec * 1e9 + end.tv_nsec) - (start.tv_sec * 1e9 + start.tv_nsec);
    double secs = nanosecs / 1e9;
    fprintf(f, "%lf\n", secs);

    freemat(A);
    freemat(B);
    freemat(C);
  }

  fclose(f);
  printf("\nDone.\n");
}
