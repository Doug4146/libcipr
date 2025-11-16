/**
 * @file blur_box.c
 *
 * Implements the separable, AVX2-optimized, multithreaded box blur filter
 * declared in LIBCIPR/libcipr.h.
 */

#include "LIBCIPR/libcipr.h"
#include "image/image.h"
#include "threading/thread_pool.h"
#include "utils/utils.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>

/**
 * @brief Executes the horizontal pass of a separable box blur using AVX2
 * intrinsics.
 *
 * Performs a 1D convolution along each row of the source image between y_start
 * and y_end using a uniform box blur filter of width `kernel_size`. The core
 * loop processes 16 pixels at a time using AVX2 vector intrinsics to execute:
 *      - Loading as u8
 *      - u8 -> i16 type conversion
 *      - Accumulation of neighboring pixel values and division by filter width
 *      - Conversion to u8 using unsigned saturation
 *      - Storing results as i16
 *
 * Boundary regions on the left and right edges of the image buffer are
 * handled with dedicated scalar fallback code.
 *
 * @param dst Pointer to the destination image buffer.
 * @param src Pointer to the source image buffer.
 * @param kernel_size Size of the separable box blur kernel.
 * @param y_start Starting row index.
 * @param y_end Ending row index.
 */
static void blur_box_horizontal_pass_avx2(cipr_u8 *dst, cipr_u8 *src, cipr_i32 kernel_size,
                                          cipr_i32 h, cipr_i32 w, cipr_i32 stride, cipr_i32 y_start,
                                          cipr_i32 y_end)
{
    // Compute box blur kernel radius
    cipr_i32 radius = kernel_size / 2;

    for (cipr_i32 y = y_start; y < y_end; y++) {

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

/**
 * @brief Executes the vertical pass of a separable box blur using AVX2
 * intrinsics.
 *
 * Performs a 1D convolution along the columns of the source image using a
 * uniform box blur filter of width `kernel_size`, operating on rows between
 * y_start and y_end. The core loop processes 16 pixels at a time using AVX2
 * vector intrinsics to execute:
 *      - Loading as u8
 *      - u8 -> i16 type conversion
 *      - Accumulation of neighboring pixel values and division by filter width
 *      - Conversion to u8 using unsigned saturation
 *      - Storing results as u8
 *
 * Boundary regions on the top and bottom edges of the image buffer are
 * handled with dedicated scalar fallback code.
 *
 * @param dst Pointer to the destination image buffer.
 * @param src Pointer to the source image buffer.
 * @param kernel_size Size of the separable box blur kernel.
 * @param y_start Starting row index.
 * @param y_end Ending row index.
 */
static void blur_box_vertical_pass_avx2(cipr_u8 *dst, cipr_u8 *src, cipr_i32 kernel_size,
                                        cipr_i32 h, cipr_i32 w, cipr_i32 stride, cipr_i32 y_start,
                                        cipr_i32 y_end)
{
    // Compute box blur kernel radius
    cipr_i32 radius = kernel_size / 2;

    cipr_i32 y = y_start;
    // ---- Upper boundary handling (non-vectorized) ----
    for (; (y < radius) && (y < y_end); y++) {

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
    for (; (y < h - radius) && (y < y_end); y++) {

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
    for (; (y < h) && (y < y_end); y++) {

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

// Argument structure for both box blur thread tasks
struct Arg_EitherPass {
    cipr_u8 *dst;
    cipr_u8 *src;
    cipr_i32 kernel_size;
    cipr_i32 h;
    cipr_i32 w;
    cipr_i32 stride;
    cipr_i32 y_start;
    cipr_i32 y_end;
};

// Thread task function for the horizontal box blur pass (AVX2)
static void *blur_box_horizontal_avx2_task(void *argument)
{
    struct Arg_EitherPass *arg = (struct Arg_EitherPass *)argument;
    blur_box_horizontal_pass_avx2(arg->dst, arg->src, arg->kernel_size, arg->h, arg->w, arg->stride,
                                  arg->y_start, arg->y_end);
    return NULL;
}

// Thread task function for the vertical box blur pass (AVX2)
static void *blur_box_vertical_avx2_task(void *argument)
{
    struct Arg_EitherPass *arg = (struct Arg_EitherPass *)argument;
    blur_box_vertical_pass_avx2(arg->dst, arg->src, arg->kernel_size, arg->h, arg->w, arg->stride,
                                arg->y_start, arg->y_end);
    return NULL;
}

/**
 * @brief Executes the multi-threaded AVX2 box blur.
 *
 * Divides the image buffers into row ranges and runs the horizontal and
 * vertical box blur passes sequentially in parallel threads, blocking until
 * each pass completes.
 *
 * @return 0 on success, -1 on failure.
 */
static int blur_box_avx2_mt(struct CIPR__PlanarView *orig_planar_view,
                            struct CIPR__PlanarView *temp_planar_view, cipr_i32 kernel_size,
                            cipr_i32 h, cipr_i32 w, cipr_i32 stride)
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

            arg_list[index].kernel_size = kernel_size;
            arg_list[index].h = h;
            arg_list[index].w = w;
            arg_list[index].stride = stride;
            arg_list[index].y_start = (h * m) / tasks_per_plane;
            arg_list[index].y_end = (h * (m + 1)) / tasks_per_plane;
        }
    }

    // Pass 1: horizontal box blur (temporary buffer <- original buffer)
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view->num_planes; n++) {
        for (cipr_i32 m = 0; m < tasks_per_plane; m++) {
            cipr_i32 index = n * tasks_per_plane + m;

            arg_list[index].dst = (cipr_u8 *)temp_planar_view->planes[n];
            arg_list[index].src = (cipr_u8 *)orig_planar_view->planes[n];

            cipr__thread_pool_submit(blur_box_horizontal_avx2_task, &arg_list[index]);
        }
    }

    // Block until horizontal pass is fully complete
    cipr__thread_pool_wait_all();

    // Pass 2: vertical box blur (original buffer <- temporary buffer)
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view->num_planes; n++) {
        for (cipr_i32 m = 0; m < tasks_per_plane; m++) {
            cipr_i32 index = n * tasks_per_plane + m;

            arg_list[index].dst = (cipr_u8 *)orig_planar_view->planes[n];
            arg_list[index].src = (cipr_u8 *)temp_planar_view->planes[n];

            cipr__thread_pool_submit(blur_box_vertical_avx2_task, &arg_list[index]);
        }
    }

    // Block until vertical pass is fully complete
    cipr__thread_pool_wait_all();

    // Free array of argument structures
    free(arg_list);

    return 0;
}

int cipr_filter_blur_box(CIPR_Image *image, int size)
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

    // Apply the box blur filter
    if (blur_box_avx2_mt(&orig_planar_view, &temp_planar_view, size, h, w, stride) != 0) {
        cipr__aligned_free(temp_buffer);
        return -1;
    }

    // Free the temporary buffer
    cipr__aligned_free(temp_buffer);

    return 0;
}