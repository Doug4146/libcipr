/**
 * @file image.h
 *
 * Internal header for the image structure, planar view structure, and their
 * related operations.
 */

#ifndef LIBCIPR_IMAGE_IMAGE_H_
#define LIBCIPR_IMAGE_IMAGE_H_

#include "LIBCIPR/libcipr.h"
#include "utils/utils.h"
#include <stdbool.h>

// Enumeration of supported pixel formats in libcipr
enum CIPR__PixelFormat { CIPR_PIXFMT_GRAY8, CIPR_PIXFMT_RGB8 };

// Definition of the CIPR_Image structure
struct CIPR_Image {
    bool is_loaded;
    enum CIPR__PixelFormat pixfmt;
    cipr_i32 height;
    cipr_i32 width;
    cipr_i32 stride; // (padded) elements per row
    cipr_u8 *buffer;
    cipr_usize buffer_length;
};

/**
 * @brief Validates a `CIPR_Image` structure.
 *
 * Checks if the fields of an image structure are valid, including checks for a
 * loaded state, valid dimensions and stride, and valid buffer length and
 * buffer pointer.
 *
 * @param image Pointer to the `CIPR_Image` structure.
 * @return 0 if valid, -1 otherwise.
 */
int cipr__image_validate(CIPR_Image *image);

/**
 * @brief Clones a `CIPR_Image` structure.
 *
 * Allocates memory for an image structure and its internal buffer and
 * initializes all fields to match those of the provided image structure.
 *
 * @param image Pointer to the `CIPR_Image` structure to be cloned.
 * @return Pointer to the cloned `CIPR_Image` structure, or NULL upon failure.
 */
CIPR_Image *cipr__image_clone(CIPR_Image *image);

/**
 * @brief Definition of the planar view structure.
 *
 * The buffer of a `CIPR_Image` structure stores pixels in a planar layout
 * (RR..GG...BB). The planar view structure provides an array of pointers to
 * each individual plane.
 *
 * For RGB8, num_planes = 3. planes[0], planes[1], planes[2] are pointers to
 * the R, G, and B planar regions respectively.
 *
 * For GRAY8, num_planes = 1. planes[0] is a pointer to the grayscale plane,
 * and planes[1], planes[2] are NULL.
 */
struct CIPR__PlanarView {
    void *planes[3];       // Plane base pointers: [plane0][plane1][plane2]
    cipr_usize num_planes; // 1 or 3 (GRAY8 or RGB8)
};

/**
 * @brief Initializes a `CIPR__PlanarView` structure.
 *
 * Initializes a planar view structure of a specified buffer with given element
 * size, pixel format, height, width and stride. Parameters are assumed to be
 * valid.
 *
 * @param planar_view Pointer to the `CIPR__PlanarView` structure to initialize.
 * @param buffer Pointer to the buffer.
 * @param element_size Size of each element in the buffer. Usually 1 for
 * cipr_u8 but may vary.
 */
void cipr__planar_view_init(struct CIPR__PlanarView *planar_view, void *buffer,
                            cipr_usize element_size, enum CIPR__PixelFormat pixfmt, cipr_i32 height,
                            cipr_i32 stride);

#endif // LIBCIPR_IMAGE_IMAGE_H_