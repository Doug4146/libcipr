# Benchmarks

This directory contains the code and sample images used for benchmarking
the various versions of the Gaussian and box blur filters implemented in
libcipr. 

The sample images are fully black JPEG files of various sizes (e.g., 1280x720, 
2560x1440, 7680x4320, etc.) created using the OpenCV library.

Each benchmark was run for 10 iterations and the minimum time in milliseconds
was documented for comparison. The benchmarks were executed on a Ryzen 9 6900HS
8-core processor running Windows 11, and compiled using the MSYS2 UCRT64 Mingw-w64
GCC toolchain with compile flags: `-O3 -march=native -mavx2 -mfma -flto`. libcipr
was configured to use 8 threads for all multithreaded benchmark results.

**Note**: *Final (Multithreaded)* represents the optimized version used internally by
libcipr. The other versions (naive, separable, AVX2, running-sum, etc.) are legacy
variants provided for comparison; they are implemented in the `src/filter/legacy/`
directory.

### Gaussian Blur (σ = 2.5)

| Function Version      | 1280x720 (ms) | 2560x1440 (ms) | 7680x4320 (ms) |
| :-------------------- | ------------: | -------------: | -------------: |
| Naive                 |   510.36      |   2064.95      |   18738.26     |
| Separable             |    49.93      |    207.23      |    1873.95     |
| Separable (AVX2)      |     6.04      |     22.85      |     199.82     |
| Final (Multithreaded) |     1.56      |      5.74      |      51.00     |


### Box Blur (kernel size = 15)

| Function Version        | 1280x720 (ms) | 2560x1440 (ms) | 7680x4320 (ms) |
| :---------------------- | ------------: | -------------: | -------------: |
| Naive                   |   309.74      |   1239.53      |   10963.34     |
| Separable               |    40.08      |    161.20      |    1455.68     |
| Running-Sum             |     7.58      |     51.30      |     416.99     |
| Running-Sum (Transpose) |     9.59      |     38.73      |     426.07     |
| Separable (AVX2)        |     2.07      |      7.26      |      62.53     |
| Final (Multithreaded)   |     0.73      |      2.33      |      22.86     |
