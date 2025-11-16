/**
 * @file format_convert.c
 *
 * Implements the pixel format conversion operations declared in
 * LIBCIPR/libcipr.h
 *
 * Note: these operations could potentially be further optimized with SIMD
 * and/or multithreading.
 */

#include "LIBCIPR/libcipr.h"
#include "threading/thread_pool.h"
#include "image/image.h"
#include "utils/utils.h"

// Applies the `average` grayscale conversion algorithm: gray = (r + b + g) / 2
static void convert_GRAY8_average(cipr_u8 *gray_buffer, cipr_u8 *r_buffer, cipr_u8 *g_buffer,
                                  cipr_u8 *b_buffer, cipr_i32 h, cipr_i32 w, cipr_i32 stride)
{
    // Apply conversion
    for (cipr_i32 y = 0; y < h; y++) {
        for (cipr_i32 x = 0; x < w; x++) {

            cipr_i32 r_value = r_buffer[y * stride + x];
            cipr_i32 g_value = g_buffer[y * stride + x];
            cipr_i32 b_value = b_buffer[y * stride + x];

            cipr_i32 gray_value = (r_value + g_value + b_value) / 3;
            gray_buffer[y * stride + x] = (cipr_u8)gray_value;
        }
    }
}

// Applies the `luminosity` grayscale conversion algorithm: gray = 0.3*r + 0.59*b + 0.11*g
static void convert_GRAY8_luminosity(cipr_u8 *gray_buffer, cipr_u8 *r_buffer, cipr_u8 *g_buffer,
                                     cipr_u8 *b_buffer, cipr_i32 h, cipr_i32 w, cipr_i32 stride)
{
    // Apply conversion
    for (cipr_i32 y = 0; y < h; y++) {
        for (cipr_i32 x = 0; x < w; x++) {

            cipr_i32 r_value = r_buffer[y * stride + x];
            cipr_i32 g_value = g_buffer[y * stride + x];
            cipr_i32 b_value = b_buffer[y * stride + x];

            cipr_i32 gray_value = (0.3 * r_value) + (0.59 * g_value) + (0.11 * b_value);
            gray_buffer[y * stride + x] = (cipr_u8)gray_value;
        }
    }
}

// Applies the `lightness` grayscale conversion algorithm: gray = (max(r,g,b) - min(r,g,b)) / 2
static void convert_GRAY8_lightness(cipr_u8 *gray_buffer, cipr_u8 *r_buffer, cipr_u8 *g_buffer,
                                    cipr_u8 *b_buffer, cipr_i32 h, cipr_i32 w, cipr_i32 stride)
{
    // Apply conversion
    for (cipr_i32 y = 0; y < h; y++) {
        for (cipr_i32 x = 0; x < w; x++) {

            cipr_i32 r_value = r_buffer[y * stride + x];
            cipr_i32 g_value = g_buffer[y * stride + x];
            cipr_i32 b_value = b_buffer[y * stride + x];

            cipr_i32 max_value = cipr__max_pair_u8(cipr__max_pair_u8(r_value, g_value), b_value);
            cipr_i32 min_value = cipr__min_pair_u8(cipr__min_pair_u8(r_value, g_value), b_value);

            cipr_i32 gray_value = (max_value - min_value) / 2;
            gray_buffer[y * stride + x] = (cipr_u8)gray_value;
        }
    }
}

int cipr_format_convert_to_GRAY8(CIPR_Image *image, CIPR_GRAY8Algorithm algorithm)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }

    // Validate parameter
    if (cipr__image_validate(image) != 0) {
        return -1;
    }

    // Check if pixel format is GRAY8
    if (image->pixfmt == CIPR_PIXFMT_GRAY8) {
        return 0;
    }

    // Shortened variable names
    cipr_i32 h = image->height;
    cipr_i32 w = image->width;
    cipr_i32 stride = image->stride;

    // Allocate memory for a planar GRAY8 buffer
    cipr_usize gray_buffer_length = image->buffer_length / 3;
    cipr_u8 *gray_buffer =
        cipr__aligned_alloc(gray_buffer_length * sizeof(cipr_u8), CIPR_CACHELINE);
    if (gray_buffer == NULL) {
        return -1;
    }

    // Initialize a planar view structure for the original RGB8 buffer
    struct CIPR__PlanarView rgb_plane_view;
    cipr__planar_view_init(&rgb_plane_view, image->buffer, sizeof(cipr_u8), CIPR_PIXFMT_RGB8, h,
                           stride);

    // Pointers to each plane in RGB8 buffer
    cipr_u8 *r_buffer = rgb_plane_view.planes[0];
    cipr_u8 *g_buffer = rgb_plane_view.planes[1];
    cipr_u8 *b_buffer = rgb_plane_view.planes[2];

    // Apply the specified conversion algorithm
    switch (algorithm) {
    case CIPR_GRAY8_AVERAGE:
        convert_GRAY8_average(gray_buffer, r_buffer, g_buffer, b_buffer, h, w, stride);
        break;
    case CIPR_GRAY8_LUMINOSITY:
        convert_GRAY8_luminosity(gray_buffer, r_buffer, g_buffer, b_buffer, h, w, stride);
        break;
    case CIPR_GRAY8_LIGHTNESS:
        convert_GRAY8_lightness(gray_buffer, r_buffer, g_buffer, b_buffer, h, w, stride);
        break;
    }

    // Free the original buffer
    cipr__aligned_free(image->buffer);

    // Update image structure fields
    image->pixfmt = CIPR_PIXFMT_GRAY8;
    image->buffer = gray_buffer;
    image->buffer_length = gray_buffer_length;

    return 0;
}

int cipr_format_convert_to_RGB8(CIPR_Image *image)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }

    // Validate parameter
    if (cipr__image_validate(image) != 0) {
        return -1;
    }

    // Check if pixel format is RGB8
    if (image->pixfmt == CIPR_PIXFMT_RGB8) {
        return 0;
    }

    // Shortened variable names
    cipr_i32 h = image->height;
    cipr_i32 w = image->width;
    cipr_i32 stride = image->stride;

    // Allocate memory for a planar RGB8 buffer
    cipr_usize rgb_buffer_length = 3 * image->buffer_length;
    cipr_u8 *rgb_buffer = cipr__aligned_alloc(rgb_buffer_length * sizeof(cipr_u8), CIPR_CACHELINE);
    if (rgb_buffer == NULL) {
        return -1;
    }

    // Initialize a planar view structure for the RGB8 buffer
    struct CIPR__PlanarView rgb_plane_view;
    cipr__planar_view_init(&rgb_plane_view, rgb_buffer, sizeof(cipr_u8), CIPR_PIXFMT_RGB8, h,
                           stride);

    // Pointers to each plane in RGB8 buffer
    cipr_u8 *r_buffer = rgb_plane_view.planes[0];
    cipr_u8 *g_buffer = rgb_plane_view.planes[1];
    cipr_u8 *b_buffer = rgb_plane_view.planes[2];

    // Apply conversion
    for (cipr_i32 y = 0; y < h; y++) {
        for (cipr_i32 x = 0; x < w; x++) {

            cipr_u8 gray_value = image->buffer[y * stride + x];

            r_buffer[y * stride + x] = gray_value;
            g_buffer[y * stride + x] = gray_value;
            b_buffer[y * stride + x] = gray_value;
        }
    }

    // Free the original buffer
    cipr__aligned_free(image->buffer);

    // Update image structure fields
    image->pixfmt = CIPR_PIXFMT_RGB8;
    image->buffer = rgb_buffer;
    image->buffer_length = rgb_buffer_length;

    return 0;
}