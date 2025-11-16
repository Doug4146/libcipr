# libcipr

High-performance image processing library written in C.

### Features
* PNG I/O (libspng)
* JPEG I/O (libjpeg-turbo)
* Internal pixel layout conversion (interleaved <-> planar)
* AVX2-optimized image filters:
    * Gaussian blur
    * Box blur
    * Unsharp mask
    * Sobel operator
* Format conversion (RGB ↔ grayscale)
* Custom internal thread-pool (Pthreads)

### Requirments
* x86-64 CPU with **AVX2** and **FMA** support
* CMake version 3.15 or newer
* GCC or Clang C compiler
* **Pthreads** (Linux/WSL; Windows supported via a Mingw-w64 toolchain)

### Building

```
git clone https://github.com/Doug4146/libcipr.git
cd libcipr
cmake -B build
cmake --build build
```
- The static library is generated at `build/libcipr.a`
- Example and benchmark executables generated in the `build/` directory

### Examples

Several example programs are included in the `examples/` directory (e.g., `eg_gaussian.c`, `eg_combined.c`, etc.). A sample image (from https://unsplash.com/photos/aerial-view-of-city-during-day-time-kCABKZBt4Gk) is provided for quick testing. 

To run an example program:
```
cd build
./eg_gaussian
```

### Benchmarks

Several benchmarking programs are included in the `benchmarks/` directory to compare performance between different versions of the Gaussian and box blur filters.

See `benchmarks/README.md` for more information and results.