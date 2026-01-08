/**
 * @file legacy_blur_box.c
 *
 * Implements the legacy box blur functions.
 */

#include "LIBCIPR/libcipr.h"
#include "image/image.h"
#include "threading/thread_pool.h"
#include "utils/utils.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>

// ----------------------------------------------------------------------------
// Naive box blur implementation
// ----------------------------------------------------------------------------

// Naive box blur (scalar, single-threaded): 2D convolution => O(k^2), k = size
static int blur_box_naive(CIPR_Image *image, int size)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }

    // Validate parameters
    if ((cipr__image_validate(image) != 0) || (size <= 0) || (size % 2 == 0)) {
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

    // Compute kernel radius for box blur
    cipr_i32 radius = size / 2;

    // Apply algorithm for each planar region of image buffer
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view.num_planes; n++) {

        cipr_u8 *orig = (cipr_u8 *)orig_planar_view.planes[n];
        cipr_u8 *temp = (cipr_u8 *)temp_planar_view.planes[n];

        for (cipr_i32 y = 0; y < h; y++) {
            for (cipr_i32 x = 0; x < w; x++) {

                cipr_i32 result = 0;

                for (cipr_i32 yy = y - radius; yy <= y + radius; yy++) {
                    for (cipr_i32 xx = x - radius; xx <= x + radius; xx++) {
                        if ((yy >= 0) && (yy < h) && (xx >= 0) && (xx < w)) {
                            result += orig[yy * stride + xx];
                        }
                    }
                }

                temp[y * stride + x] = (cipr_u8)(result / (size * size));
            }
        }
    }

    // Free original image buffer, set the pointer to the temporary buffer
    cipr__aligned_free(image->buffer);
    image->buffer = temp_buffer;

    return 0;
}

// ----------------------------------------------------------------------------
// Separable box blur implementation
// ----------------------------------------------------------------------------

// Separable box blur (scalar, single-threaded): 1D convolution => O(k), k=size
static int blur_box_separable(CIPR_Image *image, int size)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }

    // Validate parameters
    if ((cipr__image_validate(image) != 0) || (size <= 0) || (size % 2 == 0)) {
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

    // Compute kernel radius for box blur
    cipr_i32 radius = size / 2;

    // Apply algorithm for each planar region of image buffer
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view.num_planes; n++) {

        cipr_u8 *orig = (cipr_u8 *)orig_planar_view.planes[n];
        cipr_u8 *temp = (cipr_u8 *)temp_planar_view.planes[n];

        // Pass 1: horizontal box blur (temp <- orig)
        for (cipr_i32 y = 0; y < h; y++) {
            for (cipr_i32 x = 0; x < w; x++) {

                cipr_i32 result = 0;

                for (cipr_i32 xx = x - radius; xx <= x + radius; xx++) {
                    if ((xx >= 0) && (xx < w)) {
                        result += orig[y * stride + xx];
                    }
                }

                temp[y * stride + x] = (cipr_u8)(result / size);
            }
        }

        // Pass 2: vertical box blur (orig <- temp)
        for (cipr_i32 y = 0; y < h; y++) {
            for (cipr_i32 x = 0; x < w; x++) {

                cipr_i32 result = 0;

                for (cipr_i32 yy = y - radius; yy <= y + radius; yy++) {
                    if ((yy >= 0) && (yy < h)) {
                        result += temp[yy * stride + x];
                    }
                }

                orig[y * stride + x] = (cipr_u8)(result / size);
            }
        }
    }

    // Free the temporary buffer
    cipr__aligned_free(temp_buffer);

    return 0;
}

// ----------------------------------------------------------------------------
// Running-sum box blur implementation
// ----------------------------------------------------------------------------

// Running-sum box blur (scalar, single-threaded): 1D running-sum => O(1)
static int blur_box_running_sum(CIPR_Image *image, int size)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }

    // Validate parameters
    if ((cipr__image_validate(image) != 0) || (size <= 0) || (size % 2 == 0)) {
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

    // Compute kernel radius for box blur
    cipr_i32 radius = size / 2;

    // Apply algorithm for each planar region of image buffer
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view.num_planes; n++) {

        cipr_u8 *orig = (cipr_u8 *)orig_planar_view.planes[n];
        cipr_u8 *temp = (cipr_u8 *)temp_planar_view.planes[n];

        // Pass 1: horizontal running sum (temp <- orig)
        for (cipr_i32 y = 0; y < h; y++) {

            cipr_i32 left = -(radius)-1;
            cipr_i32 right = radius;

            // Initial sum computation
            cipr_i32 sum = 0;
            for (cipr_i32 i = 0; i <= right; i++) {
                sum += orig[y * stride + i];
            }

            // Running sum across row
            for (cipr_i32 x = 0; x < w; x++) {

                temp[y * stride + x] = (cipr_u8)(sum / size);

                if (++left >= 0) {
                    sum -= orig[y * stride + left];
                }
                if (++right < w) {
                    sum += orig[y * stride + right];
                }
            }
        }

        // Pass 2: vertical running sum (orig <- temp)
        for (cipr_i32 x = 0; x < w; x++) {

            cipr_i32 above = -(radius)-1;
            cipr_i32 below = radius;

            // Initial sum computation
            cipr_i32 sum = 0;
            for (cipr_i32 j = 0; j <= below; j++) {
                sum += temp[j * stride + x];
            }

            // Running sum across column
            for (cipr_i32 y = 0; y < h; y++) {

                orig[y * stride + x] = (cipr_u8)(sum / size);

                if (++above >= 0) {
                    sum -= temp[above * stride + x];
                }
                if (++below < h) {
                    sum += temp[below * stride + x];
                }
            }
        }
    }

    // Free the temporary buffer
    cipr__aligned_free(temp_buffer);

    return 0;
}

// ----------------------------------------------------------------------------
// Running-sum with transpose box blur implementation
// ----------------------------------------------------------------------------

// Tiled transpose function
static void transpose(cipr_u8 *dst, cipr_u8 *src, cipr_i32 h, cipr_i32 w, cipr_i32 stride)
{
    cipr_i32 block_size = 32; // (could try cache-unaware algorithm instead)

    // Iterate over blocks in column-major order
    for (cipr_i32 xx = 0; xx < w; xx += block_size) {
        cipr_i32 bx;
        if (xx + block_size < w) {
            bx = xx + block_size;
        } else {
            bx = w;
        }

        for (cipr_i32 yy = 0; yy < h; yy += block_size) {
            cipr_i32 by;
            if (yy + block_size < h) {
                by = yy + block_size;
            } else {
                by = h;
            }

            // Iterate over the block itself in column-major order
            for (cipr_i32 x = xx; x < bx; x++) {
                for (cipr_i32 y = yy; y < by; y++) {

                    // Swap rows and columns
                    dst[x * h + y] = src[y * stride + x];
                }
            }
        }
    }
}

// Box blur running-sum horizontal pass
static void running_sum_horizontal_pass(cipr_u8 *dst, cipr_u8 *src, cipr_i32 size, cipr_i32 h,
                                        cipr_i32 w, cipr_i32 stride)
{
    // Compute kernel radius for box blur
    cipr_i32 radius = size / 2;

    for (cipr_i32 y = 0; y < h; y++) {

        cipr_i32 left = -(radius)-1;
        cipr_i32 right = radius;

        // Initial sum computation
        cipr_i32 sum = 0;
        for (cipr_i32 i = 0; i <= right; i++) {
            sum += src[y * stride + i];
        }

        // Running sum across row
        for (cipr_i32 x = 0; x < w; x++) {

            dst[y * stride + x] = (cipr_u8)(sum / size);

            if (++left >= 0) {
                sum -= src[y * stride + left];
            }
            if (++right < w) {
                sum += src[y * stride + right];
            }
        }
    }
}

// Running-sum box blur with transpose (scalar, single-threaded)
static int blur_box_running_sum_transpose(CIPR_Image *image, int size)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }

    // Validate parameters
    if ((cipr__image_validate(image) != 0) || (size <= 0) || (size % 2 == 0)) {
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

    // Apply algorithm for each planar region of image buffer
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view.num_planes; n++) {

        cipr_u8 *orig = (cipr_u8 *)orig_planar_view.planes[n];
        cipr_u8 *temp = (cipr_u8 *)temp_planar_view.planes[n];

        // Pass 1: horizontal running sum (temp <- orig)
        running_sum_horizontal_pass(temp, orig, size, h, w, stride);

        // Transpose 1: (orig <- temp)
        transpose(orig, temp, h, w, stride);

        // Pass 2: vertical (technically horizontal) running sum (temp <- orig)
        running_sum_horizontal_pass(temp, orig, size, w, h, h);

        // Transpose 2: (orig <- temp)
        transpose(orig, temp, w, h, h);
    }

    // Free the temporary buffer
    cipr__aligned_free(temp_buffer);

    return 0;
}

// ----------------------------------------------------------------------------
// Separable box blur with AVX2 vector intrinsics
// ----------------------------------------------------------------------------

// AVX2-optimized separable horizontal box blur pass
static void separable_horizontal_avx2(cipr_u8 *dst, cipr_u8 *src, cipr_i32 kernel_size, cipr_i32 h,
                                      cipr_i32 w, cipr_i32 stride)
{
    // Compute box blur kernel radius
    cipr_i32 radius = kernel_size / 2;

    for (cipr_i32 y = 0; y < h; y++) {

        // Pointers to current row in dst and src buffers
        cipr_u8 *restrict dst_row = &dst[y * stride];
        const cipr_u8 *restrict src_row = &src[y * stride];

        cipr_i32 x = 0;
        // ---- Left boundary handling (non-vectorized) ----
        for (; x < radius; x++) {

            cipr_i32 result = 0;

            for (cipr_i32 xx = 0; xx <= x + radius; xx++) {
                result += src_row[xx];
            }
            dst_row[x] = (cipr_u8)(result / kernel_size);
        }
        // ---- Main loop (vectorized) ----
        for (; x < (w - radius) - 16; x += 16) {

            // Initialize an i16 vector with all elements set to 0
            __m256i result_i16 = _mm256_setzero_si256();

            for (cipr_i32 xx = x - radius; xx <= x + radius; xx++) {

                // Load (unaligned) sixteen u8 values from src - upconvert to i16
                __m128i src_u8 = _mm_loadu_si128((const __m128i *)&src_row[xx]);
                __m256i src_i16 = _mm256_cvtepu8_epi16(src_u8);

                // Add src_i16 to result_i16
                result_i16 = _mm256_add_epi16(src_i16, result_i16);
            }

            // Approximate integer division by 'size' using multiplication and 16-bit right shift
            __m256i multiplier_i16 = _mm256_set1_epi16((int16_t)((1 << 16) / kernel_size));
            __m256i quotient_i16 = _mm256_mulhi_epi16(
                result_i16, multiplier_i16); // Stored as 32-bit temp - extracts upper 16 bits

            // Extract lower and upper eight i16 values of result_i16
            __m128i result_lo_i16 = _mm256_extracti128_si256(quotient_i16, 0);
            __m128i result_hi_i16 = _mm256_extracti128_si256(quotient_i16, 1);

            // Downconvert i16 values to u8 (unsigned saturation), store in a single vector
            __m128i result_u8 = _mm_packus_epi16(result_lo_i16, result_hi_i16);

            // Store (unaligned) result_u8 to dst
            _mm_storeu_si128((__m128i *)&dst_row[x], result_u8);
        }
        // ---- Main loop (non-vectorized, handling leftovers) ----
        for (; x < w - radius; x++) {

            cipr_i32 result = 0;

            for (cipr_i32 xx = x - radius; xx <= x + radius; xx++) {
                result += src_row[xx];
            }
            dst_row[x] = (cipr_u8)(result / kernel_size);
        }
        // ---- Right boundary handling (non-vectorized) ----
        for (; x < w; x++) {

            cipr_i32 result = 0;

            for (cipr_i32 xx = x - radius; xx < w; xx++) {
                result += src_row[xx];
            }
            dst_row[x] = (cipr_u8)(result / kernel_size);
        }
    }
}

// AVX2-optimized separable vertical box blur pass
static void separable_vertical_avx2(cipr_u8 *dst, cipr_u8 *src, cipr_i32 kernel_size, cipr_i32 h,
                                    cipr_i32 w, cipr_i32 stride)
{
    // Compute box blur kernel radius
    cipr_i32 radius = kernel_size / 2;

    cipr_i32 y = 0;
    // ---- Upper boundary handling (non-vectorized) ----
    for (; y < radius; y++) {

        // Pointer to current row in dst buffer
        cipr_u8 *restrict dst_row = &dst[y * stride];

        for (cipr_i32 x = 0; x < w; x++) {

            cipr_i32 result = 0;

            for (cipr_i32 yy = 0; yy <= y + radius; yy++) {
                result += src[yy * stride + x];
            }
            dst_row[x] = (cipr_u8)(result / kernel_size);
        }
    }
    // ---- Main loop (vectorized) ----
    for (; y < h - radius; y++) {

        // Pointer to current row in dst buffer
        cipr_u8 *restrict dst_row = &dst[y * stride];

        cipr_i32 x = 0;
        // ---- Main loop (vectorized) ----
        for (; x < w - 16; x += 16) {

            // Initialize an i16 vector with all elements set to 0
            __m256i result_i16 = _mm256_setzero_si256();

            for (cipr_i32 yy = y - radius; yy <= y + radius; yy++) {

                // Load (aligned) sixteen u8 values from src - upconvert to i16
                __m128i src_u8 = _mm_load_si128((const __m128i *)&src[yy * stride + x]);
                __m256i src_i16 = _mm256_cvtepu8_epi16(src_u8);

                // Add src_i16 to result_i16
                result_i16 = _mm256_add_epi16(src_i16, result_i16);
            }

            // Approximate integer division by 'size' using multiplication and 16-bit right shift
            __m256i multiplier_i16 = _mm256_set1_epi16((int16_t)((1 << 16) / kernel_size));
            __m256i quotient_i16 = _mm256_mulhi_epi16(
                result_i16, multiplier_i16); // Stored as 32-bit temp - extracts upper 16 bits

            // Extract lower and upper eight i16 values of result_i16
            __m128i result_lo_i16 = _mm256_extracti128_si256(quotient_i16, 0);
            __m128i result_hi_i16 = _mm256_extracti128_si256(quotient_i16, 1);

            // Downconvert i16 values to u8 (unsigned saturation), store in a single vector
            __m128i result_u8 = _mm_packus_epi16(result_lo_i16, result_hi_i16);

            // Store (aligned) result_u8 to dst
            _mm_store_si128((__m128i *)&dst_row[x], result_u8);
        }
        // ---- Main loop (non-vectorized, handling leftovers) ----
        for (; x < w; x++) {

            cipr_i32 result = 0;

            for (cipr_i32 yy = y - radius; yy <= y + radius; yy++) {
                result += src[yy * stride + x];
            }
            dst_row[x] = (cipr_u8)(result / kernel_size);
        }
    }
    // ---- Lower boundary handling (non-vectorized) ----
    for (; y < h; y++) {

        // Pointer to current row in dst buffer
        cipr_u8 *restrict dst_row = &dst[y * stride];

        for (cipr_i32 x = 0; x < w; x++) {

            cipr_i32 result = 0;

            for (cipr_i32 yy = y - radius; yy < h; yy++) {
                result += src[yy * stride + x];
            }
            dst_row[x] = (cipr_u8)(result / kernel_size);
        }
    }
}

// AVX2-optimized separable box blur (single-threaded)
static int blur_box_separable_avx2(CIPR_Image *image, int size)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }

    // Validate parameters
    if ((cipr__image_validate(image) != 0) || (size <= 0) || (size % 2 == 0)) {
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

    // Apply algorithm for each planar region of image buffer
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view.num_planes; n++) {

        cipr_u8 *orig = (cipr_u8 *)orig_planar_view.planes[n];
        cipr_u8 *temp = (cipr_u8 *)temp_planar_view.planes[n];

        // Pass 1: horizontal box blur (temp <- orig)
        separable_horizontal_avx2(temp, orig, size, h, w, stride);

        // Pass 2: vertical box blur (orig <- temp)
        separable_vertical_avx2(orig, temp, size, h, w, stride);
    }

    // Free the temporary buffer
    cipr__aligned_free(temp_buffer);

    return 0;
}

// ----------------------------------------------------------------------------
// Public function for legacy box blur implementations
// ----------------------------------------------------------------------------

// Function table for the different gaussian blur implementations
typedef int (*blur_box_func)(CIPR_Image *, int);
blur_box_func blur_box_function_table[] = {blur_box_naive, blur_box_separable, blur_box_running_sum,
                                           blur_box_running_sum_transpose, blur_box_separable_avx2};

int cipr_legacy_filter_blur_box(CIPR_Image *image, int size, CIPR_BLUR_BOX_IMPL implementation)
{
    // Call the specified Gaussian implentation from the function table
    blur_box_func function = blur_box_function_table[implementation];
    function(image, size);

    return 0;
}