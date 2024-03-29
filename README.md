# cuexpm - Matrix Exponential Approximation using CUDA

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![GitHub Release](https://img.shields.io/github/v/release/maximilianbehr/cuexpm?display_name=release&style=flat)
 [![DOI](https://zenodo.org/badge/758500958.svg)](https://zenodo.org/doi/10.5281/zenodo.10844529)



**Version:** 2.0.0

**Copyright:** Maximilian Behr

**License:** The software is licensed under under MIT. See [`LICENSE`](LICENSE) for details.

`cuexpm` is a `CUDA` library for the numerical approximation of the matrix exponential $e^A$.

`cuexpm` supports single and double precision as well as real and complex matrices.


| Functions                        | Data                             |
| ---------------------------------|----------------------------------|
| `cuexpms_bufferSize`, `cuexpms`  | real, single precision matrix    |
| `cuexpmd_bufferSize`, `cuexpmd`  | real, double precision matrix    |
| `cuexpmc_bufferSize`, `cuexpmc`  | complex, single precision matrix |
| `cuexpmz_bufferSize`, `cuexpmz`  | complex, double precision matrix |


Available functions:

```C
int cuexpms_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cuexpms(const int n, const float *d_A, const int ldA, void *d_buffer, void *h_buffer, float *d_expmA, const int ldexpmA);
```
```C
int cuexpmd_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cuexpmd(const int n, const double *d_A, const int ldA, void *d_buffer, void *h_buffer, double *d_expmA, const int ldexpmA);
```
```C
int cuexpmc_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cuexpmc(const int n, const cuComplex *d_A, const int ldA, void *d_buffer, void *h_buffer, cuComplex *d_expmA, const int ldexpmA);
```
```C
int cuexpmz_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cuexpmz(const int n, const cuDoubleComplex *d_A, const int ldA, void *d_buffer, void *h_buffer, cuDoubleComplex *d_expmA, const int ldexpmA);
```


## Algorithm

`cuexpm` implements the scaling and squaring method for the matrix exponential approximation. 

> The Scaling and Squaring Method for the Matrix Exponential Revisited
Nicholas J. Higham
SIAM Journal on Matrix Analysis and Applications 2005 26:4, 1179-1193 


## Installation

Prerequisites:
 * `CMake >= 3.23`
 * `CUDA >= 11.4.2`

```shell
  mkdir build && cd build
  cmake ..
  make
  make install
```

## Usage and Examples

We provide examples for all supported matrix formats:

  
| File                                       | Data                             |
| -------------------------------------------|----------------------------------|
| [`example_cuexpms.cu`](example_cuexpms.cu) | real, single precision matrix    |
| [`example_cuexpmd.cu`](example_cuexpmd.cu) | real, double precision matrix    |
| [`example_cuexpmc.cu`](example_cuexpmc.cu) | complex, single precision matrix |
| [`example_cuexpmz.cu`](example_cuexpmz.cu) | complex, double precision matrix |

