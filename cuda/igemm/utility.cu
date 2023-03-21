#include <stdio.h>

#include "utility.h"
#include <iostream>

void init(int8_t *a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = rand() % 4;
  }
}

void init(float *a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = rand() % 512 / 256.f;
  }
}

// a : row
// b : col
// c: row
template <typename T>
void naive_gemm_cpu(const int8_t *a, const int8_t *b, T *c_cpu, int m,
                    int n, int k, const T* c_bias) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      T sum = 0;
      for (int ii = 0; ii < k; ii++) {
        sum += a[i * k + ii] * b[ii + j * k];
      }
      sum += c_bias[j];
      c_cpu[i * n + j] = sum;
    }
  }
}

template void naive_gemm_cpu(const int8_t *a, const int8_t *b, int32_t *c_cpu, int m, int n, int k, const int32_t* c_bias);
template void naive_gemm_cpu(const int8_t *a, const int8_t *b, float *c_cpu, int m, int n, int k, const float* c_bias);


int32_t diff(const int32_t *c, const int32_t *c_baseline, int n) {
  int32_t max_diff = -1;
  for (int i = 0; i < n; i++) {
    if (std::abs(c_baseline[i] - c[i]) > max_diff) {
      max_diff = std::abs(c_baseline[i] - c[i]);
    }
  }
  return max_diff;
}

int32_t diff(const int8_t *c, const int32_t *c_baseline, int n) {
  int32_t max_diff = -1;
  for (int i = 0; i < n; i++) {
    int8_t modify_base = c_baseline[i];
    if (c_baseline[i] >= 127) {
      modify_base = 127;
    } else if(c_baseline[i] <= -128) {
      modify_base = -128;
    }
    if (modify_base != c[i]) {
      return 10000;
    }
  }
  return max_diff;
}


float diff(const float *c, const float *c_baseline, int n) {
  float max_diff = -1;
  for (int i = 0; i < n; i++) {
    // printf("%f\n", c_baseline[i]);
    // printf("%f\n", c[i]);
    if (std::abs(c_baseline[i] - c[i]) > max_diff) {
      max_diff = std::abs(c_baseline[i] - c[i]);
    }
  }
  return max_diff;
}