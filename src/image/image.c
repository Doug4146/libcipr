/**
 * @file task_queue.c
 *
 * Implements the image structure and planar view structure handling operations
 * declared in image/image.h and LIBCIPR/libcipr.h.
 */

#include "image/image.h"
#include "threading/thread_pool.h"
#include "LIBCIPR/libcipr.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

CIPR_Image *cipr_image_create()
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return NULL;
    }

    // Allocate struct
    CIPR_Image *image = malloc(sizeof(CIPR_Image));
    if (image == NULL) {
        return NULL;
    }

    // Initialize struct fields
    image->is_loaded = false;
    image->pixfmt = CIPR_PIXFMT_RGB8;
    image->height = 0;
    image->width = 0;
    image->stride = 0;
    image->buffer = NULL;
    image->buffer_length = 0;

    return image;
}

void cipr_image_destroy(CIPR_Image **image)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return;
    }
    
    // Validate argument
    if ((image == NULL) || (*image == NULL)) {
        return;
    }

    // Free and nullify the buffer field
    if ((*image)->buffer != NULL) {
        cipr__aligned_free((*image)->buffer);
        (*image)->buffer = NULL;
    }

    // Free and nullify struct
    if (*image != NULL) {
        free(*image);
        *image = NULL;
    }
}

int cipr__image_validate(CIPR_Image *image)
{
    // Validate argument
    if (image == NULL) {
        return -1;
    }

    // Check if loaded
    if (!image->is_loaded) {
        return -1;
    }

    // Check dimensions
    if ((image->height < 1) || (image->width < 1)) {
        return -1;
    }

    // Check stride
    if (image->stride < image->width) {
        return -1;
    }

    // Check buffer
    if (image->buffer == NULL) {
        return -1;
    }

    // Check buffer length
    if (image->pixfmt == CIPR_PIXFMT_GRAY8) {
        if ((cipr_i32)image->buffer_length < (image->stride * image->height)) {
            return -1;
        }
    } else if (image->pixfmt == CIPR_PIXFMT_RGB8) {
        if ((cipr_i32)image->buffer_length < (3 * image->stride * image->height)) {
            return -1;
        }
    }

    return 0;
}

CIPR_Image *cipr__image_clone(CIPR_Image *image)
{
    // Validate parameter
    if (cipr__image_validate(image) != 0) {
        return NULL;
    }

    // Allocate memory for struct
    CIPR_Image *out_image = malloc(sizeof(CIPR_Image));
    if (out_image == NULL) {
        return NULL;
    }

    // Allocate memory for buffer
    out_image->buffer = cipr__aligned_alloc(image->buffer_length * sizeof(cipr_u8), CIPR_CACHELINE);
    if (out_image->buffer == NULL) {
        free(out_image);
        return NULL;
    }

    // Initialize struct fields
    out_image->is_loaded = true;
    out_image->pixfmt = image->pixfmt;
    out_image->height = image->height;
    out_image->width = image->width;
    out_image->stride = image->stride;
    memcpy(out_image->buffer, image->buffer, image->buffer_length);
    out_image->buffer_length = image->buffer_length;

    return out_image;
}

void cipr__planar_view_init(struct CIPR__PlanarView *planar_view, void *buffer,
                            cipr_usize element_size, enum CIPR__PixelFormat pixfmt, cipr_i32 height,
                            cipr_i32 stride)
{
    // For safe pointer arithmetic
    cipr_u8 *buffer_u8 = (cipr_u8 *)buffer;

    switch (pixfmt) {
    case CIPR_PIXFMT_GRAY8:
        planar_view->num_planes = 1;
        planar_view->planes[0] = buffer_u8;
        planar_view->planes[1] = NULL;
        planar_view->planes[2] = NULL;
        break;
    case CIPR_PIXFMT_RGB8:
        planar_view->num_planes = 3;
        planar_view->planes[0] = buffer_u8 + element_size * (0 * height * stride);
        planar_view->planes[1] = buffer_u8 + element_size * (1 * height * stride);
        planar_view->planes[2] = buffer_u8 + element_size * (2 * height * stride);
    }
}