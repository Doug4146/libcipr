/**
 * @file png_io.c
 *
 * Implements the PNG file read and write operations declared in
 * LIBCIPR/libcipr.h
 */

#include "LIBCIPR/libcipr.h"
#include "image/image.h"
#include "io/layout_convert.h"
#include "utils/utils.h"
#include "threading/thread_pool.h"
#include <spng.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

int cipr_io_read_png(CIPR_Image *image, const char *filename)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }

    // Validate parameters
    if ((image == NULL) || (filename == NULL)) {
        return -1;
    }

    // Check if image is loaded
    if (image->is_loaded) {
        return -1;
    }

    // Open binary file for reading
    FILE *png_file = fopen(filename, "rb");
    if (png_file == NULL) {
        return -1;
    }

    // Create a spng decoder context
    spng_ctx *spng_context = spng_ctx_new(0);
    if (spng_context == NULL) {
        fclose(png_file);
        return -1;
    }

    // Set source PNG file
    if (spng_set_png_file(spng_context, png_file) != 0) {
        spng_ctx_free(spng_context);
        fclose(png_file);
        return -1;
    }

    // Read PNG file header
    struct spng_ihdr png_header;
    if (spng_get_ihdr(spng_context, &png_header) != 0) {
        spng_ctx_free(spng_context);
        fclose(png_file);
        return -1;
    }

    // Determine required buffer length for decoding output (tight (non-padded) interleaved RGB8)
    cipr_usize tight_buffer_length;
    if (spng_decoded_image_size(spng_context, SPNG_FMT_RGB8, &tight_buffer_length) != 0) {
        spng_ctx_free(spng_context);
        fclose(png_file);
        return -1;
    }

    // Allocate memory for tight (non-padded) interleaved buffer
    cipr_u8 *tight_buffer = cipr__aligned_alloc(tight_buffer_length * sizeof(cipr_u8), CIPR_CACHELINE);
    if (tight_buffer == NULL) {
        spng_ctx_free(spng_context);
        fclose(png_file);
        return -1;
    }

    // Decode PNG file to the tight buffer
    if (spng_decode_image(spng_context, tight_buffer, tight_buffer_length, SPNG_FMT_RGB8, 0) != 0) {
        cipr__aligned_free(tight_buffer);
        spng_ctx_free(spng_context);
        fclose(png_file);
        return -1;
    }

    // Free spng context and close file
    spng_ctx_free(spng_context);
    fclose(png_file);

    // Determine image stride by rounding width up to a multiple of cacheline
    cipr_i32 stride = cipr__round_up(png_header.width, CIPR_CACHELINE);

    // Allocate memory for a padded buffer and zero the bytes
    cipr_usize padded_buffer_length = 3 * png_header.height * stride;
    cipr_u8 *padded_buffer = cipr__aligned_alloc(padded_buffer_length * sizeof(cipr_u8), CIPR_CACHELINE);
    if (padded_buffer == NULL) {
        cipr__aligned_free(tight_buffer);
        return -1;
    }
    memset(padded_buffer, 0, padded_buffer_length);

    // Apply padding to tight buffer and convert to planar layout
    cipr__layout_interleaved_to_planar_RGB8(padded_buffer, tight_buffer, png_header.height,
                                            png_header.width, stride);

    // Free the tight buffer
    cipr__aligned_free(tight_buffer);

    // Update image structure fields
    image->is_loaded = true;
    image->pixfmt = CIPR_PIXFMT_RGB8;
    image->height = png_header.height;
    image->width = png_header.width;
    image->stride = stride;
    image->buffer = padded_buffer;
    image->buffer_length = padded_buffer_length;

    return 0;
}

int cipr_io_write_png(CIPR_Image *image, const char *filename)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }
    
    // Validate parameters
    if ((cipr__image_validate(image) != 0) || (filename == NULL)) {
        return -1;
    }

    // Open a binary file for writing
    FILE *png_file = fopen(filename, "wb");
    if (png_file == NULL) {
        return -1;
    }

    // Create an spng encoder context
    spng_ctx *spng_context = spng_ctx_new(SPNG_CTX_ENCODER);
    if (spng_context == NULL) {
        fclose(png_file);
        remove(filename);
        return -1;
    }

    // Set PNG file
    if (spng_set_png_file(spng_context, png_file) != 0) {
        spng_ctx_free(spng_context);
        fclose(png_file);
        remove(filename);
        return -1;
    }

    // Configure PNG file header
    struct spng_ihdr png_header = {0};
    png_header.height = image->height;
    png_header.width = image->width;
    png_header.bit_depth = 8;
    switch (image->pixfmt) {
    case CIPR_PIXFMT_GRAY8:
        png_header.color_type = SPNG_COLOR_TYPE_GRAYSCALE;
        break;
    case CIPR_PIXFMT_RGB8:
        png_header.color_type = SPNG_COLOR_TYPE_TRUECOLOR;
        break;
    }

    // Set PNG file header
    if (spng_set_ihdr(spng_context, &png_header) != 0) {
        spng_ctx_free(spng_context);
        fclose(png_file);
        remove(filename);
        return -1;
    }

    // Set encoding format to match format specified in png header
    int spng_fmt = SPNG_FMT_PNG;

    // Declare a tight (non-padded) interleaved buffer
    cipr_usize tight_buffer_length = 0;
    cipr_u8 *tight_buffer = NULL;

    // Allocate memory for the tight interleaved buffer
    switch (image->pixfmt) {
    case CIPR_PIXFMT_GRAY8:
        tight_buffer_length = image->height * image->width;
        tight_buffer = cipr__aligned_alloc(tight_buffer_length * sizeof(cipr_u8), CIPR_CACHELINE);
        if (tight_buffer == NULL) {
            spng_ctx_free(spng_context);
            fclose(png_file);
            remove(filename);
            return -1;
        }
        // Convert padded GRAY8 buffer to non-padded GRAY8
        cipr__layout_unpad_GRAY8(tight_buffer, image->buffer, image->height, image->width,
                                 image->stride);
        break;
    case CIPR_PIXFMT_RGB8:
        tight_buffer_length = 3 * image->height * image->width;
        tight_buffer = cipr__aligned_alloc(tight_buffer_length * sizeof(cipr_u8), CIPR_CACHELINE);
        if (tight_buffer == NULL) {
            spng_ctx_free(spng_context);
            fclose(png_file);
            remove(filename);
            return -1;
        }
        // Convert padded planar RGB8 buffer to non-padded interleaved RGB8
        cipr__layout_planar_to_interleaved_RGB8(tight_buffer, image->buffer, image->height,
                                                image->width, image->stride);
        break;
    }

    // Encode tight buffer to PNG file
    if (spng_encode_image(spng_context, tight_buffer, tight_buffer_length, spng_fmt,
                          SPNG_ENCODE_FINALIZE) != 0) {
        cipr__aligned_free(tight_buffer);
        spng_ctx_free(spng_context);
        fclose(png_file);
        remove(filename);
        return -1;
    }

    // Free the tight buffer, spng context, and close file
    cipr__aligned_free(tight_buffer);
    spng_ctx_free(spng_context);
    fclose(png_file);

    return 0;
}