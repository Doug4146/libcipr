/**
 * @file unsharp_mask.c
 *
 * Implements the multithreaded unsharp mask filter declared in
 * LIBCIPR/libcipr.h.
 */

#include "LIBCIPR/libcipr.h"
#include "image/image.h"
#include "threading/thread_pool.h"
#include "utils/utils.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>

/**
 * @brief Executes the single-pass of an unsharp mask filter.
 *
 * Applies image sharpening to the source image buffer `orig` between the
 * rows y_start and y_end using the unsharp mask algorithm:
 *      - result = original + amount * (original - blurred)
 * The operation is performed on a per-pixel basis and AVX2 vector intrinsics
 * are not currently used. The results are stored back to the source image
 * buffer.
 *
 * @param orig Pointer to the source image buffer.
 * @param blurred Pointer to the pre-blurred image buffer.
 * @param amount Multiplier for the sharpening strength.
 * @param y_start Starting row index.
 * @param y_end Ending row index.
 */
static void unsharp_mask_pass(cipr_u8 *orig, cipr_u8 *blurred, cipr_f32 amount, cipr_i32 h,
                              cipr_i32 w, cipr_i32 stride, cipr_i32 y_start, cipr_i32 y_end)
{
    for (cipr_i32 y = y_start; y < y_end; y++) {
        for (cipr_i32 x = 0; x < w; x++) {
            cipr_i32 index = y * stride + x;

            // Algorithm: original + (amount * (original - blurred))
            cipr_i32 difference = orig[index] - blurred[index];
            cipr_i32 scaled = amount * difference;
            cipr_i32 sum = orig[index] + scaled;

            orig[index] = cipr__clamp_u8_i32(sum);
        }
    }
}

// Argument structure for unsharp mask thread tasks
struct Arg {
    cipr_u8 *orig;
    cipr_u8 *blurred;
    cipr_f32 amount;
    cipr_i32 h;
    cipr_i32 w;
    cipr_i32 stride;
    cipr_i32 y_start;
    cipr_i32 y_end;
};

// Thread task function for the single unsharp mask pass (AVX2)
static void *unsharp_mask_task(void *argument)
{
    struct Arg *arg = (struct Arg *)argument;
    unsharp_mask_pass(arg->orig, arg->blurred, arg->amount, arg->h, arg->w, arg->stride,
                      arg->y_start, arg->y_end);
    return NULL;
}

/**
 * @brief Executes the multi-threaded AVX2 unsharp mask.
 *
 * Divides the image buffers into row ranges and runs the single unsharp mask
 * pass in parallel threads, blocking until the pass completes.
 *
 * @return 0 on success, -1 on failure.
 */
static int unsharp_mask_mt(struct CIPR__PlanarView *orig_planar_view,
                           struct CIPR__PlanarView *blurred_planar_view, cipr_f32 amount,
                           cipr_i32 h, cipr_i32 w, cipr_i32 stride)
{
    // Determine number of tasks to create per image plane
    cipr_i32 num_threads = cipr__thread_pool_num_threads();
    cipr_i32 tasks_per_plane = ceil((cipr_f32)num_threads / orig_planar_view->num_planes);
    if (tasks_per_plane < 1) {
        tasks_per_plane = 1;
    }

    // Create array of argument structures
    cipr_usize arg_list_length = tasks_per_plane * orig_planar_view->num_planes;
    struct Arg *arg_list = malloc(arg_list_length * sizeof(struct Arg));
    if (arg_list == NULL) {
        return -1;
    }

    // Pre-fill invariant fields in each argument structure
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view->num_planes; n++) {
        for (cipr_i32 m = 0; m < tasks_per_plane; m++) {
            cipr_i32 index = n * tasks_per_plane + m;

            arg_list[index].orig = (cipr_u8 *)orig_planar_view->planes[n];
            arg_list[index].blurred = (cipr_u8 *)blurred_planar_view->planes[n];
            arg_list[index].amount = amount;
            arg_list[index].h = h;
            arg_list[index].w = w;
            arg_list[index].stride = stride;
            arg_list[index].y_start = (h * m) / tasks_per_plane;
            arg_list[index].y_end = (h * (m + 1)) / tasks_per_plane;
        }
    }

    // Single pass (original buffer <- blurred buffer)
    for (cipr_i32 n = 0; n < (cipr_i32)orig_planar_view->num_planes; n++) {
        for (cipr_i32 m = 0; m < tasks_per_plane; m++) {
            cipr_i32 index = n * tasks_per_plane + m;
            cipr__thread_pool_submit(unsharp_mask_task, &arg_list[index]);
        }
    }

    // Block until single pass is fully complete
    cipr__thread_pool_wait_all();

    // Free array of argument structures
    free(arg_list);

    return 0;
}

int cipr_filter_unsharp_mask(CIPR_Image *image, cipr_f32 amount, cipr_f32 standard_deviation)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }

    // Validate parameters
    if ((cipr__image_validate(image) != 0) || (amount <= 0) || (standard_deviation <= 0)) {
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

    // Clone the original image structure for blurring
    CIPR_Image *blurred_image = cipr__image_clone(image);
    if (blurred_image == NULL) {
        return -1;
    }

    // Initialize a planar view structure for the blurred image buffer
    struct CIPR__PlanarView blurred_planar_view;
    cipr__planar_view_init(&blurred_planar_view, blurred_image->buffer, sizeof(cipr_u8),
                           blurred_image->pixfmt, h, stride);

    // Apply the gaussian blur filter to the blurred image structure
    if (cipr_filter_blur_gaussian(blurred_image, standard_deviation) != 0) {
        cipr_image_destroy(&blurred_image);
        return -1;
    }

    // Apply the unsharp mask filter
    if (unsharp_mask_mt(&orig_planar_view, &blurred_planar_view, amount, h, w, stride) != 0) {
        cipr_image_destroy(&blurred_image);
    }

    // Destroy the blurred image structure
    cipr_image_destroy(&blurred_image);

    return 0;
}