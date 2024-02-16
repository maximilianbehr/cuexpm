/* MIT License
 *
 * Copyright (c) 2024 Maximilian Behr
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif
int cuexpms_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cuexpmd_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cuexpmc_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cuexpmz_bufferSize(const int n, size_t *d_bufferSize, size_t *h_bufferSize);
int cuexpms(const float *d_A, const int n, void *d_buffer, void *h_buffer, float *d_expmA);
int cuexpmd(const double *d_A, const int n, void *d_buffer, void *h_buffer, double *d_expmA);
int cuexpmc(const cuComplex *d_A, const int n, void *d_buffer, void *h_buffer, cuComplex *d_expmA);
int cuexpmz(const cuDoubleComplex *d_A, const int n, void *d_buffer, void *h_buffer, cuDoubleComplex *d_expmA);
#ifdef __cplusplus
}
#endif
