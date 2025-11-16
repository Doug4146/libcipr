/**
 * @file sobel.c
 *
 * Implements the separable, AVX2-optimized, multithreaded Sobel operator
 * filter declared in LIBCIPR/libcipr.h.
 *
 * Note: Need to look into the "white" borders produced.
 */

#include "LIBCIPR/libcipr.h"
#include "image/image.h"
#include "threading/thread_pool.h"
#include "utils/utils.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>

#define SEPARABLE_KERNEL_SIZE   3
#define SEPARABLE_KERNEL_RADIUS 1

/*
 * Sobel X kernel (2D):
 *
 *      [-1  0  1]
 *      [-2  0  2]
 *      [-1  0  1]
 */
cipr_i32 kernel_X_1D_horiz[3] = {-1, 0, 1};
cipr_i32 kernel_X_1D_vert[3] = {1, 2, 1};

/*
 * Sobel Y kernel (2D):
 *
 *      [ 1   2   1]
 *      [ 0   0   0]
 *      [-1  -2  -1]
 */
cipr_i32 kernel_Y_1D_horiz[3] = {1, 2, 1};
cipr_i32 kernel_Y_1D_vert[3] = {-1, 0, 1};

/**
 * @brief Executes the horizontal pass of a separable Sobel X and Y operator
 * using AVX2 intrinsics.
 *
 * Convolves rows of the source image buffer `orig` between y_start and y_end
 * with the Sobel X and Y horizontal kernels, and storing the results in `Gx`
 * and `Gy`, respectively. The core loop processes 16 pixels at a time using
 * AVX2 vector intrinsics to execute:
 *      - Loading as u8
 *      - u8 -> i16 type conversion
 *      - Multiplication with each kernel and accumulation
 *      - Storing results as i16
 *
 * Boundary regions on the left and right edges of the image buffer are
 * handled with dedicated scalar fallback code.
 *
 * @param Gx Pointer to the intermediate image buffer for Sobel X results.
 * @param Gy Pointer to the intermediate image buffer for Sobel Y results.
 * @param orig Pointer to the source image buffer.
 * @param y_start Starting row index.
 * @param y_end Ending row index.
 */
static void sobel_horizontal_pass_avx2(cipr_i16 *Gx, cipr_i16 *Gy, cipr_u8 *orig, cipr_i32 h,
                                       cipr_i32 w, cipr_i32 stride, cipr_i32 y_start,
                                       cipr_i32 y_end)
{
    // Set kernels and set kernel radius
    cipr_i32 *Gx_kernel = kernel_X_1D_horiz;
    cipr_i32 *Gy_kernel = kernel_Y_1D_horiz;
    cipr_i32 radius = SEPARABLE_KERNEL_RADIUS;

    for (cipr_i32 y = y_start; y < y_end; y++) {

        // Pointers to current row in each buffer
        cipr_i16 *restrict Gx_row = &Gx[y * stride];
        cipr_i16 *restrict Gy_row = &Gy[y * stride];
        const cipr_u8 *restrict orig_row = &orig[y * stride];

        cipr_i32 x = 0;
        // ---- Left boundary handling (non-vectorized) ----
        for (; x < radius; x++) {

            cipr_i32 Gx_result = 0;
            cipr_i32 Gy_result = 0;

            cipr_i32 kernel_index = radius - x;
            for (cipr_i32 xx = 0; xx <= x + radius; xx++) {
                Gx_result += orig_row[xx] * Gx_kernel[kernel_index];
                Gy_result += orig_row[xx] * Gy_kernel[kernel_index];
                kernel_index++;
            }
            Gx_row[x] = (cipr_i16)Gx_result;
            Gy_row[x] = (cipr_i16)Gy_result;
        }
        // ---- Main loop (vectorized) ----
        for (; x < (w - radius) - 16; x += 16) {

            // Initialize two i16 vectors with all elements set to 0
            __m256i Gx_result_i16 = _mm256_setzero_si256();
            __m256i Gy_result_i16 = _mm256_setzero_si256();

            cipr_i32 kernel_index = 0;

            for (cipr_i32 xx = x - radius; xx <= x + radius; xx++) {

                // Load (unaligned) sixteen u8 values from src - upconvert to i16
                __m128i src_u8 = _mm_loadu_si128((const __m128i *)&orig_row[xx]);
                __m256i src_i16 = _mm256_cvtepu8_epi16(src_u8);

                // Initialize i16 vectors with all elements set to the kernel values
                __m256i Gx_kernel_value_i16 = _mm256_set1_epi16(Gx_kernel[kernel_index]);
                __m256i Gy_kernel_value_i16 = _mm256_set1_epi16(Gy_kernel[kernel_index]);

                // Multiply kernel value and source vectors
                __m256i Gx_product_i16 = _mm256_mullo_epi16(src_i16, Gx_kernel_value_i16);
                __m256i Gy_product_i16 = _mm256_mullo_epi16(src_i16, Gy_kernel_value_i16);

                // Add product to result vectors
                Gx_result_i16 = _mm256_add_epi16(Gx_result_i16, Gx_product_i16);
                Gy_result_i16 = _mm256_add_epi16(Gy_result_i16, Gy_product_i16);

                kernel_index++;
            }

            // Store (unaligned) result_i16 vectors to Gx and Gy
            _mm256_storeu_si256((__m256i *)&(Gx_row[x]), Gx_result_i16);
            _mm256_storeu_si256((__m256i *)&(Gy_row[x]), Gy_result_i16);
        }
        // ---- Main loop (non-vectorized, handling leftovers) ----
        for (; x < w - radius; x++) {

            cipr_i32 Gx_result = 0;
            cipr_i32 Gy_result = 0;

            cipr_i32 kernel_index = 0;
            for (cipr_i32 xx = x - radius; xx <= x + radius; xx++) {
                Gx_result += orig_row[xx] * Gx_kernel[kernel_index];
                Gy_result += orig_row[xx] * Gy_kernel[kernel_index];
                kernel_index++;
            }
            Gx_row[x] = (cipr_i16)Gx_result;
            Gy_row[x] = (cipr_i16)Gy_result;
        }
        // ---- Right boundary handling (non-vectorized) ----
        for (; x < w; x++) {

            cipr_i32 Gx_result = 0;
            cipr_i32 Gy_result = 0;

            cipr_i32 kernel_index = 0;
            for (cipr_i32 xx = x - radius; xx < w; xx++) {
                Gx_result += orig_row[xx] * Gx_kernel[kernel_index];
                Gy_result += orig_row[xx] * Gy_kernel[kernel_index];
                kernel_index++;
            }
            Gx_row[x] = (cipr_i16)Gx_result;
            Gy_row[x] = (cipr_i16)Gy_result;
        }
    }
}

/**
 * @brief Executes the vertical pass of a separable Sobel X and Y operator and
 * computes the magnitude using AVX2 intrinsics.
 *
 * Convolves columns of the intermediate Sobel X and Y buffers `Gx` and
 * `Gy` with the Sobel X and Y vertical kernels, operating on rows between
 * y_start and y_end. The results are combined to determine the magnitude,
 * which is stored back to the source image buffer `orig`. The core loop
 * processes 16 pixels at a time using AVX2 vector intrinsics to execute:
 *     - Loading as i16
 *     - Multiplication with each kernel and accumulation
 *     - Magnitude determination
 *     - Conversion to u8 using unsigned saturation
 *     - Storing results as u8
 *
 * Boundary regions on the top and bottom edges of the image buffer are
 * handled with dedicated scalar fallback code.
 *
 * @param Gx Pointer to the intermediate image buffer for Sobel X results.
 * @param Gy Pointer to the intermediate image buffer for Sobel Y results.
 * @param orig Pointer to the source image buffer.
 * @param y_start Starting row index.
 * @param y_end Ending row index.
 */
static void sobel_vertical_and_magnitude_pass_avx2(cipr_i16 *Gx, cipr_i16 *Gy, cipr_u8 *orig,
                                                   cipr_i32 h, cipr_i32 w, cipr_i32 stride,
                                                   cipr_i32 y_start, cipr_i32 y_end)
{
    // Set kernels and compute kernel radius
    cipr_i32 *Gx_kernel = kernel_X_1D_vert;
    cipr_i32 *Gy_kernel = kernel_Y_1D_vert;
    cipr_i32 radius = SEPARABLE_KERNEL_RADIUS;

    cipr_i32 y = y_start;
    // ---- Upper boundary handling (non-vectorized) ----
    for (; (y < radius) && (y < y_end); y++) {

        // Pointer to current row in orig buffer
        cipr_u8 *restrict orig_row = &orig[y * stride];

        for (cipr_i32 x = 0; x < w; x++) {

            cipr_i32 Gx_result = 0;
            cipr_i32 Gy_result = 0;

            cipr_i32 kernel_index = radius - y;
            for (cipr_i32 yy = 0; yy <= y + radius; yy++) {
                Gx_result += Gx[yy * stride + x] * Gx_kernel[kernel_index];
                Gy_result += Gy[yy * stride + x] * Gy_kernel[kernel_index];
                kernel_index++;
            }
            orig_row[x] = cipr__clamp_u8_i32(abs(Gx_result) + abs(Gy_result));
        }
    }
    // ---- Main loop ----
    for (; (y < h - radius) && (y < y_end); y++) {

        // Pointer to current row in orig buffer
        cipr_u8 *restrict orig_row = &orig[y * stride];

        cipr_i32 x = 0;
        // ---- Main loop (vectorized) ----
        for (; x < w - 16; x += 16) {

            // Initialize two i16 vectors with all elements set to 0
            __m256i Gx_result_i16 = _mm256_setzero_si256();
            __m256i Gy_result_i16 = _mm256_setzero_si256();

            cipr_i32 kernel_index = 0;

            for (cipr_i32 yy = y - radius; yy <= y + radius; yy++) {

                // Load (aligned) sixteen i16 values from Gx and Gy
                __m256i Gx_src_i16 = _mm256_load_si256((const __m256i *)&Gx[yy * stride + x]);
                __m256i Gy_src_i16 = _mm256_load_si256((const __m256i *)&Gy[yy * stride + x]);

                // Initialize i16 vectors with all elements set to the kernel values
                __m256i Gx_kernel_value_i16 = _mm256_set1_epi16(Gx_kernel[kernel_index]);
                __m256i Gy_kernel_value_i16 = _mm256_set1_epi16(Gy_kernel[kernel_index]);

                // Multiply kernel value and source vectors
                __m256i Gx_product_i16 = _mm256_mullo_epi16(Gx_src_i16, Gx_kernel_value_i16);
                __m256i Gy_product_i16 = _mm256_mullo_epi16(Gy_src_i16, Gy_kernel_value_i16);

                // Add product to result vectors
                Gx_result_i16 = _mm256_add_epi16(Gx_result_i16, Gx_product_i16);
                Gy_result_i16 = _mm256_add_epi16(Gy_result_i16, Gy_product_i16);

                kernel_index++;
            }

            // Apply absolute value to each element in the vectors
            __m256i Gx_abs_i16 = _mm256_abs_epi16(Gx_result_i16);
            __m256i Gy_abs_i16 = _mm256_abs_epi16(Gy_result_i16);

            // Sum the registers
            __m256i sum_i16 = _mm256_add_epi16(Gx_abs_i16, Gy_abs_i16);

            // Extract lower and upper eight i16 values of sum_i16
            __m128i result_lo_i16 = _mm256_extracti128_si256(sum_i16, 0);
            __m128i result_hi_i16 = _mm256_extracti128_si256(sum_i16, 1);

            // Downconvert i16 values to u8 (unsigned saturation), store in a single vector
            __m128i result_u8 = _mm_packus_epi16(result_lo_i16, result_hi_i16);

            // Store (aligned) result_u8 to orig
            _mm_store_si128((__m128i *)&orig_row[x], result_u8);
        }
        // ---- Main loop (non-vectorized, handling leftovers) ----
        for (; x < w; x++) {

            cipr_i32 Gx_result = 0;
            cipr_i32 Gy_result = 0;

            cipr_i32 kernel_index = 0;
            for (cipr_i32 yy = y - radius; yy <= y + radius; yy++) {
                Gx_result += Gx[yy * stride + x] * Gx_kernel[kernel_index];
                Gy_result += Gy[yy * stride + x] * Gy_kernel[kernel_index];
                kernel_index++;
            }
            orig_row[x] = cipr__clamp_u8_i32(abs(Gx_result) + abs(Gy_result));
        }
    }
    // ---- Lower boundary handling (non-vectorized) ----
    for (; (y < h) && (y < y_end); y++) {

        // Pointer to current row in orig buffer
        cipr_u8 *restrict orig_row = &orig[y * stride];

        for (cipr_i32 x = 0; x < w; x++) {

            cipr_i32 Gx_result = 0;
            cipr_i32 Gy_result = 0;

            cipr_i32 kernel_index = 0;
            for (cipr_i32 yy = y - radius; yy < h; yy++) {
                Gx_result += Gx[yy * stride + x] * Gx_kernel[kernel_index];
                Gy_result += Gy[yy * stride + x] * Gy_kernel[kernel_index];
                kernel_index++;
            }

            orig_row[x] = cipr__clamp_u8_i32(abs(Gx_result) + abs(Gy_result));
        }
    }
}

// Argument structure for both Sobel operator thread tasks
struct Arg_EitherPass {
    cipr_i16 *Gx_buffer;
    cipr_i16 *Gy_buffer;
    cipr_u8 *orig_buffer;
    cipr_i32 h;
    cipr_i32 w;
    cipr_i32 stride;
    cipr_i32 y_start;
    cipr_i32 y_end;
};

// Thread task function for the horizontal Sobel operator pass (AVX2)
static void *sobel_horizontal_avx2_task(void *argument)
{
    struct Arg_EitherPass *arg = (struct Arg_EitherPass *)argument;
    sobel_horizontal_pass_avx2(arg->Gx_buffer, arg->Gy_buffer, arg->orig_buffer, arg->h, arg->w,
                               arg->stride, arg->y_start, arg->y_end);
    return NULL;
}

// Thread task function for the vertical Sobel operator pass (AVX2)
static void *sobel_vertical_and_magnitude_avx2_task(void *argument)
{
    struct Arg_EitherPass *arg = (struct Arg_EitherPass *)argument;
    sobel_vertical_and_magnitude_pass_avx2(arg->Gx_buffer, arg->Gy_buffer, arg->orig_buffer, arg->h,
                                           arg->w, arg->stride, arg->y_start, arg->y_end);
    return NULL;
}

/**
 * @brief Executes the multi-threaded AVX2 Sobel operator.
 *
 * Divides the image buffers into row ranges and runs the horizontal and
 * vertical (and magnitude) Sobel operator passes sequentially in parallel
 * threads, blocking until each pass completes.
 *
 * @return 0 on success, -1 on failure.
 */
static int sobel_operator_avx2_mt(cipr_i16 *Gx_buffer, cipr_i16 *Gy_buffer, cipr_u8 *orig_buffer,
                                  cipr_i32 h, cipr_i32 w, cipr_i32 stride)
{
    // Determine number of tasks to create (recall each buffer has one planar region)
    cipr_i32 num_threads = cipr__thread_pool_num_threads();
    cipr_i32 total_tasks = num_threads;

    // Create array of argument structures used for both passes
    cipr_usize arg_list_length = total_tasks;
    struct Arg_EitherPass *arg_list = malloc(arg_list_length * sizeof(struct Arg_EitherPass));
    if (arg_list == NULL) {
        return -1;
    }

    // Pre-fill invariant fields in each argument structure
    for (cipr_i32 m = 0; m < total_tasks; m++) {
        cipr_i32 index = m;

        arg_list[index].Gx_buffer = Gx_buffer;
        arg_list[index].Gy_buffer = Gy_buffer;
        arg_list[index].orig_buffer = orig_buffer;
        arg_list[index].h = h;
        arg_list[index].w = w;
        arg_list[index].stride = stride;
        arg_list[index].y_start = (h * m) / total_tasks;
        arg_list[index].y_end = (h * (m + 1)) / total_tasks;
    }

    // Pass 1: horizontal sobel X and Y ((Gx_buffer, Gy_buffer) <- orig_buffer)
    for (cipr_i32 m = 0; m < total_tasks; m++) {
        cipr_i32 index = m;
        cipr__thread_pool_submit(sobel_horizontal_avx2_task, &arg_list[index]);
    }

    // Block until horizontal pass is fully complete
    cipr__thread_pool_wait_all();

    // Pass 2: vertical sobel X and Y and magnitude (orig_buffer <- (Gx_buffer, Gy_buffer))
    for (cipr_i32 m = 0; m < total_tasks; m++) {
        cipr_i32 index = m;
        cipr__thread_pool_submit(sobel_vertical_and_magnitude_avx2_task, &arg_list[index]);
    }

    // Block until vertical pass is fully complete
    cipr__thread_pool_wait_all();

    // Free array of argument structures
    free(arg_list);

    return 0;
}

int cipr_filter_sobel_operator(CIPR_Image *image)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }

    // Validate parameters
    if ((cipr__image_validate(image) != 0) || (image->pixfmt != CIPR_PIXFMT_GRAY8)) {
        return -1;
    }

    // Shortened variable names
    cipr_i32 h = image->height;
    cipr_i32 w = image->width;
    cipr_i32 stride = (cipr_u32)image->stride;

    // Allocate memory for an intermediate buffer for Gx
    cipr_i16 *Gx_buffer =
        cipr__aligned_alloc(image->buffer_length * sizeof(cipr_i16), CIPR_CACHELINE);
    if (Gx_buffer == NULL) {
        return -1;
    }

    // Allocate memory for an intermediate buffer for Gy
    cipr_i16 *Gy_buffer =
        cipr__aligned_alloc(image->buffer_length * sizeof(cipr_i16), CIPR_CACHELINE);
    if (Gy_buffer == NULL) {
        cipr__aligned_free(Gx_buffer);
        return -1;
    }

    // Apply the Sobel operator filter
    if (sobel_operator_avx2_mt(Gx_buffer, Gy_buffer, image->buffer, h, w, stride) != 0) {
        cipr__aligned_free(Gy_buffer);
        cipr__aligned_free(Gx_buffer);
        return -1;
    }

    // Free the intermediate buffers
    cipr__aligned_free(Gy_buffer);
    cipr__aligned_free(Gx_buffer);

    return 0;
}