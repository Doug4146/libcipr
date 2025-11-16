/**
 * @file jpeg_io.c
 *
 * Implements the JPEG file read and write operations declared in
 * LIBCIPR/libcipr.h
 */

#include "LIBCIPR/libcipr.h"
#include "image/image.h"
#include "io/layout_convert.h"
#include "utils/utils.h"
#include "threading/thread_pool.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <turbojpeg.h>

int cipr_io_read_jpeg(CIPR_Image *image, const char *filename)
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
    FILE *jpeg_file = fopen(filename, "rb");
    if (jpeg_file == NULL) {
        return -1;
    }

    // Determine size of the file
    if (fseek(jpeg_file, 0, SEEK_END) != 0) {
        fclose(jpeg_file);
        return -1;
    }
    cipr_long jpeg_file_size = ftell(jpeg_file);
    if (jpeg_file_size < 0) {
        fclose(jpeg_file);
        return -1;
    }
    if (fseek(jpeg_file, 0, SEEK_SET) != 0) {
        fclose(jpeg_file);
        return -1;
    }

    // Create a buffer for the file data
    cipr_u8 *jpeg_file_buffer = (cipr_u8 *)malloc(jpeg_file_size * sizeof(cipr_u8));
    if (jpeg_file_buffer == NULL) {
        fclose(jpeg_file);
        return -1;
    }

    // Read the file data into the buffer
    if (fread(jpeg_file_buffer, 1, jpeg_file_size, jpeg_file) != (cipr_usize)jpeg_file_size) {
        free(jpeg_file_buffer);
        fclose(jpeg_file);
        return -1;
    }

    // Close file
    fclose(jpeg_file);

    // Create a jpeg decompressing instance
    tjhandle tj_instance = tj3Init(TJINIT_DECOMPRESS);
    if (tj_instance == NULL) {
        free(jpeg_file_buffer);
        return -1;
    }

    // Read JPEG file header
    if (tj3DecompressHeader(tj_instance, jpeg_file_buffer, jpeg_file_size) != 0) {
        tj3Destroy(tj_instance);
        free(jpeg_file_buffer);
        return -1;
    }
    int height = tj3Get(tj_instance, TJPARAM_JPEGHEIGHT);
    int width = tj3Get(tj_instance, TJPARAM_JPEGWIDTH);

    // Allocate memory for tight (non-padded) interleaved RGB8 buffer
    cipr_usize tight_buffer_length = 3 * height * width;
    cipr_u8 *tight_buffer =
        cipr__aligned_alloc(tight_buffer_length * sizeof(cipr_u8), CIPR_CACHELINE);
    if (tight_buffer == NULL) {
        tj3Destroy(tj_instance);
        free(jpeg_file_buffer);
        return -1;
    }

    // Decompress JPEG file to the tight buffer only if precision <= 8 bits
    int precision = tj3Get(tj_instance, TJPARAM_PRECISION);
    if (precision > 8) {
        cipr__aligned_free(tight_buffer);
        tj3Destroy(tj_instance);
        free(jpeg_file_buffer);
        return -1;
    }
    if (tj3Decompress8(tj_instance, jpeg_file_buffer, jpeg_file_size, tight_buffer, 0, TJPF_RGB) !=
        0) {
        cipr__aligned_free(tight_buffer);
        tj3Destroy(tj_instance);
        free(jpeg_file_buffer);
        return -1;
    }

    // Destroy jpeg decompressing instance and free jpeg file buffer
    tj3Destroy(tj_instance);
    free(jpeg_file_buffer);

    // Determine image stride by rounding width up to a multiple of cacheline
    cipr_usize stride = cipr__round_up(width, CIPR_CACHELINE);

    // Allocate memory for a padded buffer and zero the bytes
    cipr_usize padded_buffer_length = 3 * height * stride;
    cipr_u8 *padded_buffer =
        cipr__aligned_alloc(padded_buffer_length * sizeof(cipr_u8), CIPR_CACHELINE);
    if (padded_buffer == NULL) {
        cipr__aligned_free(tight_buffer);
        return -1;
    }
    memset(padded_buffer, 0, padded_buffer_length);

    // Apply padding to tight buffer and convert to planar layout
    cipr__layout_interleaved_to_planar_RGB8(padded_buffer, tight_buffer, height, width, stride);

    // Free the tight buffer
    cipr__aligned_free(tight_buffer);

    // Update image structure fields
    image->is_loaded = true;
    image->pixfmt = CIPR_PIXFMT_RGB8;
    image->height = height;
    image->width = width;
    image->stride = stride;
    image->buffer = padded_buffer;
    image->buffer_length = padded_buffer_length;

    return 0;
}

int cipr_io_write_jpeg(CIPR_Image *image, const char *filename, int quality)
{
    // Check if the global thread pool is initialized
    if (!cipr__thread_pool_is_init()) {
        return -1;
    }
    
    // Validate parameters
    if ((cipr__image_validate(image) != 0) || (filename == NULL) || (quality < 1) ||
        (quality > 100)) {
        return -1;
    }

    // Open a binary file for writing
    FILE *jpeg_file = fopen(filename, "wb");
    if (jpeg_file == NULL) {
        return -1;
    }

    // Create a jpeg compressing instance
    tjhandle tj_instance = tj3Init(TJINIT_COMPRESS);
    if (tj_instance == NULL) {
        fclose(jpeg_file);
        remove(filename);
        return -1;
    }

    // Non-intuitive but required step
    int chroma_subsamp;
    switch (image->pixfmt) {
    case CIPR_PIXFMT_GRAY8:
        chroma_subsamp = TJSAMP_GRAY;
        break;
    case CIPR_PIXFMT_RGB8:
        chroma_subsamp = TJSAMP_444;
        break;
    }
    if (tj3Set(tj_instance, TJPARAM_SUBSAMP, chroma_subsamp) != 0) {
        tj3Destroy(tj_instance);
        fclose(jpeg_file);
        remove(filename);
        return -1;
    }

    // Set the JPEG compressing quality
    if (tj3Set(tj_instance, TJPARAM_QUALITY, quality) != 0) {
        tj3Destroy(tj_instance);
        fclose(jpeg_file);
        remove(filename);
        return -1;
    }

    // Declare a tight (non-padded) interleaved buffer
    cipr_usize tight_buffer_length = 0;
    cipr_u8 *tight_buffer = NULL;

    // Allocate memory for the tight interleaved buffer
    switch (image->pixfmt) {
    case CIPR_PIXFMT_GRAY8:
        tight_buffer_length = image->height * image->width;
        tight_buffer = cipr__aligned_alloc(tight_buffer_length * sizeof(cipr_u8), CIPR_CACHELINE);
        if (tight_buffer == NULL) {
            tj3Destroy(tj_instance);
            fclose(jpeg_file);
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
            tj3Destroy(tj_instance);
            fclose(jpeg_file);
            remove(filename);
            return -1;
        }
        // Convert padded planar RGB8 buffer to non-padded interleaved RGB8
        cipr__layout_planar_to_interleaved_RGB8(tight_buffer, image->buffer, image->height,
                                                image->width, image->stride);
        break;
    }

    // Set compressing format
    int jpeg_format;
    switch (image->pixfmt) {
    case CIPR_PIXFMT_GRAY8:
        jpeg_format = TJPF_GRAY;
        break;
    case CIPR_PIXFMT_RGB8:
        jpeg_format = TJPF_RGB;
        break;
    }

    // Declare empty buffer for compressing output
    cipr_usize jpeg_file_size = 0;
    cipr_u8 *jpeg_file_buffer = NULL;

    // Compress tight buffer to JPEG buffer
    if (tj3Compress8(tj_instance, tight_buffer, image->width, 0, image->height, jpeg_format,
                     &jpeg_file_buffer, &jpeg_file_size) != 0) {
        cipr__aligned_free(tight_buffer);
        tj3Destroy(tj_instance);
        fclose(jpeg_file);
        remove(filename);
        return -1;
    }

    // Free the tight buffer
    cipr__aligned_free(tight_buffer);

    // Write jpeg buffer to JPEG file
    if (fwrite(jpeg_file_buffer, jpeg_file_size, 1, jpeg_file) < 1) {
        tj3Free(jpeg_file_buffer);
        tj3Destroy(tj_instance);
        fclose(jpeg_file);
        remove(filename);
        return -1;
    }

    // Free jpeg buffer, destroy jpeg compressing instance and close file
    tj3Free(jpeg_file_buffer);
    tj3Destroy(tj_instance);
    fclose(jpeg_file);

    return 0;
}