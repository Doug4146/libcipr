/**
 * @file blur_gaussian.c
 *
 * Implements the separable, AVX2-optimized, multithreaded Gaussian blur
 * filter declared in LIBCIPR/libcipr.h.
 */

#include "LIBCIPR/libcipr.h"
#include "image/image.h"
#include "threading/thread_pool.h"
#include "utils/utils.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>

// Computes Gaussian kernel size from a given standard deviation
static cipr_u32 blur_gaussian_kernel_size(cipr_f32 standard_deviation)
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

/**
 * Initializes a 1D separable Gaussian kernel from a given kernel size and
 * standard deviation and applies normalization. The separable gaussian kernel
 * is symmetric, so the same kernel is used for both the horizontal and
 * vertical passes.
 */
static void blur_gaussian_kernel_1D_init(cipr_f32 *kernel, cipr_i32 kernel_size,
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

/**
 * @brief Executes the horizontal pass of a separable Gaussian blur using AVX2
 * intrinsics.
 *
 * Convolves rows of the source image buffer between y_start and y_end with a
 * 1D separable Gaussian kernel. The core loop processes 16 pixels at a time
 * using AVX2 vector intrinsics to execute:
 *      - Loading as u8
 *      - u8 -> i16 -> i32 -> f32 type conversion
 *      - Fused-multiply add (FMA) accumulation with the kernel
 *      - Conversion to u8 using unsigned saturation
 *      - Storing results as u8
 *
 * Boundary regions on the left and right edges of the image buffer are
 * handled with dedicated scalar fallback code.
 *
 * @param dst Pointer to the destination image buffer.
 * @param src Pointer to the source image buffer.
 * @param kernel Pointer to the 1D separable Gaussian kernel.
 * @param kernel_size Size of the 1D separable Gaussian kernel.
 * @param y_start Starting row index.
 * @param y_end Ending row index.
 */
static void blur_gaussian_horizontal_pass_avx2(cipr_u8 *dst, cipr_u8 *src, cipr_f32 *kernel,
                                               cipr_i32 kernel_size, cipr_i32 h, cipr_i32 w,
                                               cipr_i32 stride, cipr_i32 y_start, cipr_i32 y_end)
{
    // Compute gaussian kernel radius
    cipr_i32 radius = kernel_size / 2;

    for (cipr_i32 y = y_start; y < y_end; y++) {

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

/**
 * @brief Executes the vertical pass of a separable Gaussian blur using AVX2
 * intrinsics.
 *
 * Convolves columns of the source image buffer with 1D separable Gaussian
 * kernel, operating on rows between y_start and y_end. The core loop processes
 * 16 pixels at a time using AVX2 vector intrinsics to execute:
 *      - Loading as u8
 *      - u8 -> i16 -> i32 -> f32 type conversion
 *      - Fused-multiply add (FMA) accumulation with the kernel
 *      - Conversion to u8 using unsigned saturation
 *      - Storing results as u8
 *
 * Boundary regions on the top and bottom edges of the image buffer are
 * handled with dedicated scalar fallback code.
 *
 * @param dst Pointer to the destination image buffer.
 * @param src Pointer to the source image buffer.
 * @param kernel Pointer to the 1D separable Gaussian kernel.
 * @param kernel_size Size of the 1D separable Gaussian kernel.
 * @param y_start Starting row index.
 * @param y_end Ending row index.
 */
static void blur_gaussian_vertical_pass_avx2(cipr_u8 *dst, cipr_u8 *src, cipr_f32 *kernel,
                                             cipr_i32 kernel_size, cipr_i32 h, cipr_i32 w,
                                             cipr_i32 stride, cipr_i32 y_start, cipr_i32 y_end)
{
    // Compute gaussian kernel radius
    cipr_i32 radius = kernel_size / 2;

    cipr_i32 y = y_start;
    // ---- Upper boundary handling (non-vectorized) ----
    for (; (y < radius) && (y < y_end); y++) {

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
    for (; (y < h - radius) && (y < y_end); y++) {

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
    for (; (y < h) && (y < y_end); y++) {

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

// Argument structure for both Gaussian blur thread tasks
struct Arg_EitherPass {
    cipr_u8 *dst;
    cipr_u8 *src;
    cipr_f32 *kernel;
    cipr_i32 kernel_size;
    cipr_i32 h;
    cipr_i32 w;
    cipr_i32 stride;
    cipr_i32 y_start;
    cipr_i32 y_end;
};

// Thread task function for the horizontal Gaussian blur pass (AVX2)
static void *blur_gaussian_horizontal_avx2_task(void *argument)
{
    struct Arg_EitherPass *arg = (struct Arg_EitherPass *)argument;
    blur_gaussian_horizontal_pass_avx2(arg->dst, arg->src, arg->kernel, arg->kernel_size, arg->h,
                                       arg->w, arg->stride, arg->y_start, arg->y_end);
    return NULL;
}

// Thread task function for the vertical Gaussian blur pass (AVX2)
static void *blur_gaussian_vertical_avx2_task(void *argument)
{
    struct Arg_EitherPass *arg = (struct Arg_EitherPass *)argument;
    blur_gaussian_vertical_pass_avx2(arg->dst, arg->src, arg->kernel, arg->kernel_size, arg->h,
                                     arg->w, arg->stride, arg->y_start, arg->y_end);
    return NULL;
}

/**
 * @brief Executes the multi-threaded AVX2-optimized Gaussian blur.
 *
 * Divides the image buffers into row ranges and runs the horizontal and
 * vertical Gaussian blur passes sequentially in parallel threads, blocking
 * until each pass completes.
 *
 * @return 0 on success, -1 on failure.
 */
static int blur_gaussian_avx2_mt(struct CIPR__PlanarView *orig_planar_view,
                                 struct CIPR__PlanarView *temp_planar_view, cipr_f32 *kernel,
                                 cipr_i32 kernel_size, cipr_i32 h, cipr_i32 w, cipr_i32 stride)
{
    // Determine number of tasks to create per image plane
    cipr_i32 num_threads = cipr__thread_pool_num_threads();
    cipr_i32 tasks_per_plane = ceil((cipr_f32)num_threads / orig_planar_view->num_planes);
    if (tasks_per_plane < 1) {
        tasks_per_plane = 1;
    }

    // Create array of argument structures used for both passes
    cipr_usize arg_list_length = tasks_per_plane * orig_planar_view->num_planes;
    struct Arg_EitherPass *arg_list = malloc(arg_list_length * sizeof(struct Arg_EitherPass));
    if (arg_list == NULL) {
        return -1;
    }

    // Pre-fill invariant fields in each argument structure
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view->num_planes; n++) {
        for (cipr_i32 m = 0; m < tasks_per_plane; m++) {
            cipr_i32 index = n * tasks_per_plane + m;

            arg_list[index].kernel = kernel;
            arg_list[index].kernel_size = kernel_size;
            arg_list[index].h = h;
            arg_list[index].w = w;
            arg_list[index].stride = stride;
            arg_list[index].y_start = (h * m) / tasks_per_plane;
            arg_list[index].y_end = (h * (m + 1)) / tasks_per_plane;
        }
    }

    // Pass 1: horizontal gaussian blur (temporary buffer <- original buffer)
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view->num_planes; n++) {
        for (cipr_i32 m = 0; m < tasks_per_plane; m++) {
            cipr_i32 index = n * tasks_per_plane + m;

            arg_list[index].dst = (cipr_u8 *)temp_planar_view->planes[n];
            arg_list[index].src = (cipr_u8 *)orig_planar_view->planes[n];

            cipr__thread_pool_submit(blur_gaussian_horizontal_avx2_task, &arg_list[index]);
        }
    }

    // Block until horizontal pass is fully complete
    cipr__thread_pool_wait_all();

    // Pass 2: vertical gaussian blur (original buffer <- temporary buffer)
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view->num_planes; n++) {
        for (cipr_i32 m = 0; m < tasks_per_plane; m++) {
            cipr_i32 index = n * tasks_per_plane + m;

            arg_list[index].dst = (cipr_u8 *)orig_planar_view->planes[n];
            arg_list[index].src = (cipr_u8 *)temp_planar_view->planes[n];

            cipr__thread_pool_submit(blur_gaussian_vertical_avx2_task, &arg_list[index]);
        }
    }

    // Block until vertical pass is fully complete
    cipr__thread_pool_wait_all();

    // Free array of argument structures
    free(arg_list);

    return 0;
}

int cipr_filter_blur_gaussian(CIPR_Image *image, cipr_f32 standard_deviation)
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
    cipr_usize kernel_size = blur_gaussian_kernel_size(standard_deviation);
    cipr_f32 *kernel = malloc(kernel_size * sizeof(cipr_f32));
    if (kernel == NULL) {
        cipr__aligned_free(temp_buffer);
        return -1;
    }

    // Initialize the Gaussian kernel
    blur_gaussian_kernel_1D_init(kernel, kernel_size, standard_deviation);

    // Apply the Gaussian blur filter
    if (blur_gaussian_avx2_mt(&orig_planar_view, &temp_planar_view, kernel, kernel_size, h, w,
                              stride) != 0) {
        free(kernel);
        cipr__aligned_free(temp_buffer);
        return -1;
    }

    // Free the kernel and temporary buffer
    free(kernel);
    cipr__aligned_free(temp_buffer);

    return 0;
}