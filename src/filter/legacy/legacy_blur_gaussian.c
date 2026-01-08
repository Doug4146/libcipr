/**
 * @file legacy_blur_gaussian.c
 *
 * Implements the legacy gaussian blur functions.
 */

#include "LIBCIPR/libcipr.h"
#include "image/image.h"
#include "threading/thread_pool.h"
#include "utils/utils.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>

// Computes Gaussian kernel size from a given standard deviation
static cipr_u32 gaussian_kernel_size(cipr_f32 standard_deviation)
{
    // Kernel size of 6 * standard deviation captures most of Gaussian
    cipr_u32 kernel_size = ceil(6 * standard_deviation);

    // Kernel size may not be even
    if (kernel_size % 2 == 0) {
        kernel_size++;
    }

    // Minimum kernel size is 3
    if (kernel_size < 3) {
        kernel_size = 3;
    }

    return kernel_size;
}

// ----------------------------------------------------------------------------
// Naive Gaussian blur implementation
// ----------------------------------------------------------------------------

/*
 * Initializes a 2D Gaussian kernel from a given kernel size and standard
 * deviation and applies normalization.
 */
static void gaussian_kernel_2D_init(cipr_f32 *kernel, cipr_i32 kernel_size,
                                    cipr_f32 standard_deviation)
{
    // Precompute values
    cipr_f32 coefficient = 1 / (2 * CIPR__PI * standard_deviation * standard_deviation);
    cipr_f32 denominator = 2 * standard_deviation * standard_deviation;
    cipr_i32 radius = kernel_size / 2;

    cipr_f32 kernel_sum = 0;
    cipr_i32 kernel_index = 0;

    // Fill the kernel
    for (cipr_i32 y = -radius; y <= radius; y++) {
        for (cipr_i32 x = -radius; x <= radius; x++) {

            cipr_f32 value = coefficient * expf(-(x * x + y * y) / denominator);
            kernel[kernel_index] = value;

            kernel_sum += value;
            kernel_index++;
        }
    }

    // Normalize the kernel
    for (cipr_i32 i = 0; i < kernel_size * kernel_size; i++) {
        kernel[i] /= kernel_sum;
    }
}

// Naive Gaussian blur (scalar, single-threaded): 2D convolution => O(k^2), k = size
static int gaussian_naive(CIPR_Image *image, float standard_deviation)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }

    // Validate parameters
    if ((cipr__image_validate(image) != 0) || (standard_deviation <= 0)) {
        return -1;
    }

    // Shortened variable names
    cipr_i32 h = image->height;
    cipr_i32 w = image->width;
    cipr_i32 stride = (cipr_u32)image->stride;

    // Initialize a planar view structure for the original image buffer
    struct CIPR__PlanarView orig_planar_view;
    cipr__planar_view_init(&orig_planar_view, image->buffer, sizeof(cipr_u8), image->pixfmt, h,
                           stride);

    // Allocate memory for a temporary buffer
    cipr_u8 *temp_buffer =
        cipr__aligned_alloc(image->buffer_length * sizeof(cipr_u8), CIPR_CACHELINE);
    if (temp_buffer == NULL) {
        return -1;
    }

    // Initialize a planar view structure for the temporary buffer
    struct CIPR__PlanarView temp_planar_view;
    cipr__planar_view_init(&temp_planar_view, temp_buffer, sizeof(cipr_u8), image->pixfmt, h,
                           stride);

    // Allocate memory for a two-dimensional Gaussian kernel
    cipr_usize kernel_size = gaussian_kernel_size(standard_deviation);
    cipr_f32 *kernel = malloc((kernel_size * kernel_size) * sizeof(cipr_f32));
    if (kernel == NULL) {
        cipr__aligned_free(temp_buffer);
        return -1;
    }

    // Initialize the Gaussian kernel
    gaussian_kernel_2D_init(kernel, kernel_size, standard_deviation);

    // Compute gaussian kernel radius
    cipr_i32 radius = kernel_size / 2;

    // Apply algorithm for each planar region of image buffer
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view.num_planes; n++) {

        cipr_u8 *orig = (cipr_u8 *)orig_planar_view.planes[n];
        cipr_u8 *temp = (cipr_u8 *)temp_planar_view.planes[n];

        for (cipr_i32 y = 0; y < h; y++) {
            for (cipr_i32 x = 0; x < w; x++) {

                cipr_f32 result = 0;

                cipr_i32 kernel_index = 0;
                for (cipr_i32 yy = y - radius; yy <= y + radius; yy++) {
                    for (cipr_i32 xx = x - radius; xx <= x + radius; xx++) {
                        if ((yy >= 0) && (yy < h) && (xx >= 0) && (xx < w)) {
                            result += orig[yy * stride + xx] * kernel[kernel_index];
                        }
                        kernel_index++;
                    }
                }

                temp[y * stride + x] = (cipr_u8)result;
            }
        }
    }

    // Free the kernel
    free(kernel);

    // Free original image buffer, set the pointer to the temporary buffer
    cipr__aligned_free(image->buffer);
    image->buffer = temp_buffer;

    return 0;
}

// ----------------------------------------------------------------------------
// Separable Gaussian blur implementation
// ----------------------------------------------------------------------------

/**
 * Initializes a 1D separable Gaussian kernel from a given kernel size and
 * standard deviation and applies normalization. The separable gaussian kernel
 * is symmetric, so the same kernel is used for both the horizontal and
 * vertical passes.
 */
static void gaussian_kernel_1D_init(cipr_f32 *kernel, cipr_i32 kernel_size,
                                    cipr_f32 standard_deviation)
{
    // Precompute values
    cipr_f32 coefficient = 1 / (sqrt(2 * CIPR__PI) * standard_deviation);
    cipr_f32 denominator = 2 * standard_deviation * standard_deviation;
    cipr_i32 radius = kernel_size / 2;

    cipr_f32 kernel_sum = 0;
    cipr_i32 kernel_index = 0;

    // Fill the kernel
    for (cipr_i32 x = -radius; x <= radius; x++) {

        cipr_f32 value = coefficient * expf(-(x * x) / denominator);
        kernel[kernel_index] = value;

        kernel_sum += value;
        kernel_index++;
    }

    // Normalize the kernel
    for (cipr_i32 i = 0; i < kernel_size; i++) {
        kernel[i] /= kernel_sum;
    }
}

// Separable Gaussian blur (scalar, single-threaded): 1D convolution => O(k), k=size
static int gaussian_separable(CIPR_Image *image, float standard_deviation)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }

    // Validate parameters
    if ((cipr__image_validate(image) != 0) || (standard_deviation <= 0)) {
        return -1;
    }

    // Shortened variable names
    cipr_i32 h = image->height;
    cipr_i32 w = image->width;
    cipr_i32 stride = (cipr_u32)image->stride;

    // Initialize a planar view structure for the original image buffer
    struct CIPR__PlanarView orig_planar_view;
    cipr__planar_view_init(&orig_planar_view, image->buffer, sizeof(cipr_u8), image->pixfmt, h,
                           stride);

    // Allocate memory for a temporary buffer
    cipr_u8 *temp_buffer =
        cipr__aligned_alloc(image->buffer_length * sizeof(cipr_u8), CIPR_CACHELINE);
    if (temp_buffer == NULL) {
        return -1;
    }

    // Initialize a planar view structure for the temporary buffer
    struct CIPR__PlanarView temp_planar_view;
    cipr__planar_view_init(&temp_planar_view, temp_buffer, sizeof(cipr_u8), image->pixfmt, h,
                           stride);

    // Allocate memory for a one-dimensional Gaussian kernel
    cipr_usize kernel_size = gaussian_kernel_size(standard_deviation);
    cipr_f32 *kernel = malloc(kernel_size * sizeof(cipr_f32));
    if (kernel == NULL) {
        cipr__aligned_free(temp_buffer);
        return -1;
    }

    // Initialize the Gaussian kernel
    gaussian_kernel_1D_init(kernel, kernel_size, standard_deviation);

    // Compute Gaussian kernel radius
    cipr_i32 radius = kernel_size / 2;

    // Apply algorithm for each planar region of image buffer
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view.num_planes; n++) {

        cipr_u8 *orig = (cipr_u8 *)orig_planar_view.planes[n];
        cipr_u8 *temp = (cipr_u8 *)temp_planar_view.planes[n];

        // Pass 1: horizontal Gaussian blur (temp <- orig)
        for (cipr_i32 y = 0; y < h; y++) {
            for (cipr_i32 x = 0; x < w; x++) {

                cipr_f32 result = 0;

                cipr_i32 kernel_index = 0;
                for (cipr_i32 xx = x - radius; xx <= x + radius; xx++) {
                    if ((xx >= 0) && (xx < w)) {
                        result += orig[y * stride + xx] * kernel[kernel_index];
                    }
                    kernel_index++;
                }

                temp[y * stride + x] = (cipr_u8)result;
            }
        }

        // Pass 2: vertical Gaussian blur (orig <- temp)
        for (cipr_i32 y = 0; y < h; y++) {
            for (cipr_i32 x = 0; x < w; x++) {

                cipr_f32 result = 0;

                cipr_i32 kernel_index = 0;
                for (cipr_i32 yy = y - radius; yy <= y + radius; yy++) {
                    if ((yy >= 0) && (yy < h)) {
                        result += temp[yy * stride + x] * kernel[kernel_index];
                    }
                    kernel_index++;
                }

                orig[y * stride + x] = (cipr_u8)result;
            }
        }
    }

    // Free the kernel and the temporary buffer
    free(kernel);
    cipr__aligned_free(temp_buffer);

    return 0;
}

// ----------------------------------------------------------------------------
// Separable Gaussian blur with AVX2 vector intrinsics
// ----------------------------------------------------------------------------

// AVX2-optimized separable horizontal Gaussian blur pass
static void separable_horizontal_avx2(cipr_u8 *dst, cipr_u8 *src, cipr_f32 *kernel,
                                      cipr_i32 kernel_size, cipr_i32 h, cipr_i32 w, cipr_i32 stride)
{
    // Compute gaussian kernel radius
    cipr_i32 radius = kernel_size / 2;

    for (cipr_i32 y = 0; y < h; y++) {

        // Pointers to current row in dst and src buffers
        cipr_u8 *restrict dst_row = &dst[y * stride];
        const cipr_u8 *restrict src_row = &src[y * stride];

        cipr_i32 x = 0;
        // ---- Left boundary handling (non-vectorized) ----
        for (; x < radius; x++) {

            cipr_f32 result = 0;

            cipr_i32 kernel_index = radius - x;
            for (cipr_i32 xx = 0; xx <= x + radius; xx++) {
                result += src_row[xx] * kernel[kernel_index];
                kernel_index++;
            }
            dst_row[x] = (cipr_u8)result;
        }
        // ---- Main loop (vectorized) ----
        for (; x < (w - radius) - 16; x += 16) {

            // Initialize two f32 vectors with all elements set to 0
            __m256 result_lo_f32 = _mm256_setzero_ps();
            __m256 result_hi_f32 = _mm256_setzero_ps();

            cipr_i32 kernel_index = 0;

            for (cipr_i32 xx = x - radius; xx <= x + radius; xx++) {

                // Load (unaligned) sixteen u8 values from src - upconvert to i16
                __m128i src_u8 = _mm_loadu_si128((const __m128i *)&src_row[xx]);
                __m256i src_i16 = _mm256_cvtepu8_epi16(src_u8);

                // Extract lower and upper eight i16 values of src_i16
                __m128i src_lo_i16 = _mm256_extracti128_si256(src_i16, 0);
                __m128i src_hi_i16 = _mm256_extracti128_si256(src_i16, 1);

                // Upconvert i16 values to i32
                __m256i src_lo_i32 = _mm256_cvtepi16_epi32(src_lo_i16);
                __m256i src_hi_i32 = _mm256_cvtepi16_epi32(src_hi_i16);

                // Convert i32 values to f32
                __m256 src_lo_f32 = _mm256_cvtepi32_ps(src_lo_i32);
                __m256 src_hi_f32 = _mm256_cvtepi32_ps(src_hi_i32);

                // Initialize an f32 vector with all elements set to the kernel value
                __m256 kernel_value_f32 = _mm256_set1_ps(kernel[kernel_index]);

                // Multiply kernel value and source vectors, add to result vectors
                result_lo_f32 = _mm256_fmadd_ps(src_lo_f32, kernel_value_f32, result_lo_f32);
                result_hi_f32 = _mm256_fmadd_ps(src_hi_f32, kernel_value_f32, result_hi_f32);

                kernel_index++;
            }

            // Convert f32 values to i32
            __m256i result_lo_i32 = _mm256_cvtps_epi32(result_lo_f32);
            __m256i result_hi_i32 = _mm256_cvtps_epi32(result_hi_f32);

            // Downconvert i32 values to i16 - store in a single vector and shuffle
            __m256i result_i16 = _mm256_packs_epi32(
                result_lo_i32, result_hi_i32); // (a1,a2,a3,a4,b1,b2,b3,b4,a5,a6,a7,a8,b5,b6,b7,b8)
            result_i16 = _mm256_permute4x64_epi64(
                result_i16,
                _MM_SHUFFLE(3, 1, 2, 0)); // (a1,a2,a3,a4,a5,a6,a7,a8,b1,b2,b3,b4,b5,b6,b7,b8)

            // Extract lower and upper eight i16 values of result_i16
            __m128i result_lo_i16 = _mm256_extracti128_si256(result_i16, 0);
            __m128i result_hi_i16 = _mm256_extracti128_si256(result_i16, 1);

            // Downconvert i16 values to u8 (unsigned saturation), store in a single vector
            __m128i result_u8 = _mm_packus_epi16(result_lo_i16, result_hi_i16);

            // Store (unaligned) result_u8 to dst
            _mm_storeu_si128((__m128i *)&dst_row[x], result_u8);
        }
        // ---- Main loop (non-vectorized, handling leftovers) ----
        for (; x < w - radius; x++) {

            cipr_f32 result = 0;

            cipr_i32 kernel_index = 0;
            for (cipr_i32 xx = x - radius; xx <= x + radius; xx++) {
                result += src_row[xx] * kernel[kernel_index];
                kernel_index++;
            }
            dst_row[x] = (cipr_u8)result;
        }
        // ---- Right boundary handling (non-vectorized) ----
        for (; x < w; x++) {

            cipr_f32 result = 0;

            cipr_i32 kernel_index = 0;
            for (cipr_i32 xx = x - radius; xx < w; xx++) {
                result += src_row[xx] * kernel[kernel_index];
                kernel_index++;
            }
            dst_row[x] = (cipr_u8)result;
        }
    }
}

// AVX2-optimized separable vertical Gaussian box blur pass
static void separable_vertical_avx2(cipr_u8 *dst, cipr_u8 *src, cipr_f32 *kernel,
                                    cipr_i32 kernel_size, cipr_i32 h, cipr_i32 w, cipr_i32 stride)
{
    // Compute gaussian kernel radius
    cipr_i32 radius = kernel_size / 2;

    cipr_i32 y = 0;
    // ---- Upper boundary handling (non-vectorized) ----
    for (; y < radius; y++) {

        // Pointer to current row in dst buffer
        cipr_u8 *restrict dst_row = &dst[y * stride];

        for (cipr_i32 x = 0; x < w; x++) {

            cipr_f32 result = 0;

            cipr_i32 kernel_index = radius - y;
            for (cipr_i32 yy = 0; yy <= y + radius; yy++) {
                result += src[yy * stride + x] * kernel[kernel_index];
                kernel_index++;
            }
            dst_row[x] = (cipr_u8)result;
        }
    }
    // ---- Main loop ----
    for (; y < h - radius; y++) {

        // Pointer to current row in dst buffer
        cipr_u8 *restrict dst_row = &dst[y * stride];

        cipr_i32 x = 0;
        // ---- Main loop (vectorized) ----
        for (; x < w - 16; x += 16) {

            // Initialize two f32 vectors with all elements set to 0
            __m256 result_lo_f32 = _mm256_setzero_ps();
            __m256 result_hi_f32 = _mm256_setzero_ps();

            cipr_i32 kernel_index = 0;

            for (cipr_i32 yy = y - radius; yy <= y + radius; yy++) {

                // Load (aligned) sixteen u8 values from src - upconvert to i16
                __m128i src_u8 = _mm_load_si128((const __m128i *)&(src[yy * stride + x]));
                __m256i src_i16 = _mm256_cvtepu8_epi16(src_u8);

                // Extract lower and upper eight i16 values of src_i16
                __m128i src_lo_i16 = _mm256_extracti128_si256(src_i16, 0);
                __m128i src_hi_i16 = _mm256_extracti128_si256(src_i16, 1);

                // Upconvert i16 values to i32
                __m256i src_lo_i32 = _mm256_cvtepi16_epi32(src_lo_i16);
                __m256i src_hi_i32 = _mm256_cvtepi16_epi32(src_hi_i16);

                // Convert i32 values to f32
                __m256 src_lo_f32 = _mm256_cvtepi32_ps(src_lo_i32);
                __m256 src_hi_f32 = _mm256_cvtepi32_ps(src_hi_i32);

                // Initialize an f32 vector with all elements set to the kernel value
                __m256 kernel_value_f32 = _mm256_set1_ps(kernel[kernel_index]);

                // Multiply kernel value and source vectors, add to result vectors
                result_lo_f32 = _mm256_fmadd_ps(src_lo_f32, kernel_value_f32, result_lo_f32);
                result_hi_f32 = _mm256_fmadd_ps(src_hi_f32, kernel_value_f32, result_hi_f32);

                kernel_index++;
            }

            // Convert f32 values to i32
            __m256i result_lo_i32 = _mm256_cvtps_epi32(result_lo_f32);
            __m256i result_hi_i32 = _mm256_cvtps_epi32(result_hi_f32);

            // Downconvert i32 values to i16 - store in a single vector and shuffle
            __m256i result_i16 = _mm256_packs_epi32(
                result_lo_i32, result_hi_i32); // (a1,a2,a3,a4,b1,b2,b3,b4,a5,a6,a7,a8,b5,b6,b7,b8)
            result_i16 = _mm256_permute4x64_epi64(
                result_i16,
                _MM_SHUFFLE(3, 1, 2, 0)); // (a1,a2,a3,a4,a5,a6,a7,a8,b1,b2,b3,b4,b5,b6,b7,b8)

            // Extract lower and upper eight i16 values of result_i16
            __m128i result_lo_i16 = _mm256_extracti128_si256(result_i16, 0);
            __m128i result_hi_i16 = _mm256_extracti128_si256(result_i16, 1);

            // Downconvert i16 values to u8 (unsigned saturation), store in a single vector
            __m128i result_u8 = _mm_packus_epi16(result_lo_i16, result_hi_i16);

            // Store (aligned) result_u8 to dst
            _mm_store_si128((__m128i *)&dst_row[x], result_u8);
        }
        // ---- Main loop (non-vectorized, handling leftovers) ----
        for (; x < w; x++) {

            cipr_f32 result = 0;

            cipr_i32 kernel_index = 0;
            for (cipr_i32 yy = y - radius; yy <= y + radius; yy++) {
                result += src[yy * stride + x] * kernel[kernel_index];
                kernel_index++;
            }
            dst_row[x] = (cipr_u8)result;
        }
    }
    // ---- Lower boundary handling (non-vectorized) ----
    for (; y < h; y++) {

        // Pointer to current row in dst buffer
        cipr_u8 *restrict dst_row = &dst[y * stride];

        for (cipr_i32 x = 0; x < w; x++) {

            cipr_f32 result = 0;

            cipr_i32 kernel_index = 0;
            for (cipr_i32 yy = y - radius; yy < h; yy++) {
                result += src[yy * stride + x] * kernel[kernel_index];
                kernel_index++;
            }
            dst_row[x] = (cipr_u8)result;
        }
    }
}

// AVX2-Optimized separable Gaussian blur (single-threaded)
static int gaussian_separable_avx2(CIPR_Image *image, float standard_deviation)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }

    // Validate parameters
    if ((cipr__image_validate(image) != 0) || (standard_deviation <= 0)) {
        return -1;
    }

    // Shortened variable names
    cipr_i32 h = image->height;
    cipr_i32 w = image->width;
    cipr_i32 stride = (cipr_u32)image->stride;

    // Initialize a planar view structure for the original image buffer
    struct CIPR__PlanarView orig_planar_view;
    cipr__planar_view_init(&orig_planar_view, image->buffer, sizeof(cipr_u8), image->pixfmt, h,
                           stride);

    // Allocate memory for a temporary buffer
    cipr_u8 *temp_buffer =
        cipr__aligned_alloc(image->buffer_length * sizeof(cipr_u8), CIPR_CACHELINE);
    if (temp_buffer == NULL) {
        return -1;
    }

    // Initialize a planar view structure for the temporary buffer
    struct CIPR__PlanarView temp_planar_view;
    cipr__planar_view_init(&temp_planar_view, temp_buffer, sizeof(cipr_u8), image->pixfmt, h,
                           stride);

    // Allocate memory for a one-dimensional Gaussian kernel
    cipr_usize kernel_size = gaussian_kernel_size(standard_deviation);
    cipr_f32 *kernel = malloc(kernel_size * sizeof(cipr_f32));
    if (kernel == NULL) {
        cipr__aligned_free(temp_buffer);
        return -1;
    }

    // Initialize the Gaussian kernel
    gaussian_kernel_1D_init(kernel, kernel_size, standard_deviation);

    // Apply algorithm for each planar region of image buffer
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view.num_planes; n++) {

        cipr_u8 *orig = (cipr_u8 *)orig_planar_view.planes[n];
        cipr_u8 *temp = (cipr_u8 *)temp_planar_view.planes[n];

        // Pass 1: horizontal Gaussian blur (temp <- orig)
        separable_horizontal_avx2(temp, orig, kernel, kernel_size, h, w, stride);

        // Pass 2: vertical Gaussian blur (orig <- temp)
        separable_vertical_avx2(orig, temp, kernel, kernel_size, h, w, stride);
    }

    // Free the kernel and the temporary buffer
    free(kernel);
    cipr__aligned_free(temp_buffer);

    return 0;
}

// ----------------------------------------------------------------------------
// Public function for legacy Gaussian blur implementations
// ----------------------------------------------------------------------------

// Function table for the different gaussian blur implementations
typedef int (*gaussian_func)(CIPR_Image *, float);
gaussian_func gaussian_function_table[] = {gaussian_naive, gaussian_separable,
                                           gaussian_separable_avx2};

int cipr_legacy_filter_blur_gaussian(CIPR_Image *image, float standard_deviation,
                                     CIPR_GAUSSIAN_IMPL implementation)
{
    // Call the specified Gaussian implentation from the function table
    gaussian_func function = gaussian_function_table[implementation];
    function(image, standard_deviation);

    return 0;
}