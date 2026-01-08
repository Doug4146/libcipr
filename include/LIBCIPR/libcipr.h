/**
 * @file libcipr.h
 * @brief Public header for the libcipr image processing library.
 *
 * This header provides the top-level public interface for the libcipr library,
 * including threading initialization/shutdown, image structure handling, file
 * input/output operations, image format conversion operations and image
 * filtering operations.
 *
 * This header should be included to access the functionality of libcipr.
 *
 * @date 2025-11-15
 */

#ifndef LIBCIPR_LIBCIPR_H_
#define LIBCIPR_LIBCIPR_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Cacheline size in bytes (default = 64).
 *
 * Used internally for aligned memory allocation and image buffer padding. May
 * be overriden by defining CIPR_CACHELINE before including LIBCIPR/libcipr.h.
 * CIPR_CACHELINE must be a positive, power of two value.
 */
#ifndef CIPR_CACHELINE
#define CIPR_CACHELINE 64
#endif

// Compile-time check for an invalid CIPR_CACHELINE value
#if (CIPR_CACHELINE <= 0) || ((CIPR_CACHELINE & (CIPR_CACHELINE - 1)) != 0)
#error "CIPR_CACHELINE must be a positive, power of two value."
#endif

/**
 * @brief Initializes the internal thread pool of libcipr.
 *
 * Initializes the thread pool used throughout libcipr with `num_threads`
 * threads. It is recommended to create as many threads as there are cores on
 * the machine. The thread pool must be initialized prior to using any other
 * feature of libcipr.
 *
 * @param num_threads Number of threads to use for the thread pool ( >= 1).
 * @return 0 on success, -1 on fail.
 */
int cipr_thread_pool_init(int num_threads);

/**
 * @brief Releases the resources for the internal thread pool of libcipr.
 *
 * Joins all of the created threads and frees all the memory allocated for the
 * thread pool of libcipr. The thread pool must be initialized again prior to
 * using any other feature of libcipr.
 */
void cipr_thread_pool_shutdown(void);

// Opaque handle for the image structure
typedef struct CIPR_Image CIPR_Image;

/**
 * @brief Creates a `CIPR_Image` structure.
 *
 * Allocates memory for an image structure and initializes all fields. The
 * structure should eventually be destroyed with `cipr_image_destroy`.
 *
 * @return Pointer to tne `CIPR_Image` structure, or NULL upon failure.
 */
CIPR_Image *cipr_image_create();

/**
 * @brief Destroys a `CIPR_Image` structure.
 *
 * Frees all memory allocated for an image structure, including its internal
 * buffer. The provided pointer is set to NULL.
 *
 * @param image Double pointer to the `CIPR_Image` structure.
 */
void cipr_image_destroy(CIPR_Image **image);

/**
 * @brief Reads a PNG file into a `CIPR_Image` structure.
 *
 * Decodes a PNG file specified by `filename` into RGB8 pixel format and
 * initializes the specified image structure, which must be non-loaded prior to
 * calling this function.
 *
 * @param image Pointer to the `CIPR_Image` structure.
 * @param filename Path of the PNG file to read.
 * @return 0 on success, -1 on failure.
 */
int cipr_io_read_png(CIPR_Image *image, const char *filename);

/**
 * @brief Writes a `CIPR_Image` structure into a PNG file.
 *
 * Encodes the specified image structure in either GRAY8 or RGB8 pixel format
 * and writes it to a PNG file at the specified `filename`. The image structure
 * must be loaded prior to calling this function.
 *
 * @param image Pointer to the `CIPR_Image` structure.
 * @param filename Path to the output PNG file.
 * @return 0 on success, -1 on failure.
 */
int cipr_io_write_png(CIPR_Image *image, const char *filename);

/**
 * @brief Reads a JPEG file into a `CIPR_Image` structure.
 *
 * Decompresses a JPEG file specified by `filename` into RGB8 pixel format and
 * initializes the specified image structure, which must be non-loaded prior to
 * calling this function.
 *
 * @param image Pointer to the `CIPR_Image` structure.
 * @param filename Path of the JPEG file to read.
 * @return 0 on success, -1 on failure.
 */
int cipr_io_read_jpeg(CIPR_Image *image, const char *filename);

/**
 * @brief Writes a `CIPR_Image` structure into a JPEG file.
 *
 * Compresses the specified image structure in either GRAY8 or RGB8 pixel
 * format with a specified compression `quality`, and writes it to a JPEG file
 * at the specified `filename`. The image structure must be loaded prior to
 * calling this function.
 *
 * @param image Pointer to the `CIPR_Image` structure.
 * @param filename Path to the output JPEG file.
 * @param quality JPEG compression quality. Must be between 1-100.
 * @return 0 on success, -1 on failure.
 */
int cipr_io_write_jpeg(CIPR_Image *image, const char *filename, int quality);

// Enumeration of supported algorithms for grayscale conversion.
typedef enum CIPR_GRAY8Algorithm {
    CIPR_GRAY8_AVERAGE,    // gray = (r + b + g) / 3
    CIPR_GRAY8_LUMINOSITY, // gray = 0.3*r + 0.59*b + 0.11*g
    CIPR_GRAY8_LIGHTNESS   // gray = (max(r,g,b) - min(r,g,b)) / 2
} CIPR_GRAY8Algorithm;

/**
 * @brief Converts a `CIPR_Image` structure from RGB8 to GRAY8 pixel format.
 *
 * Converts the pixel data of an image structure from RGB8 to GRAY8 in-place
 * using the specified grayscale conversion `algorithm`. If image is already in
 * GRAY8 pixel format, the function returns successfully.
 *
 * @param image Pointer to the `CIPR_Image` structure.
 * @param algorithm Grayscale conversion algorithm
 * @return 0 on success, -1 on failure.
 */
int cipr_format_convert_to_GRAY8(CIPR_Image *image, CIPR_GRAY8Algorithm algorithm);

/**
 * @brief Converts a `CIPR_Image` structure from GRAY8 to RGB8 pixel format.
 *
 * Converts the pixel data of an image structure from GRAY8 to RGB8 in-place
 * by broadcasting the single-channel grayscale value across each RGB channel
 * for every pixel. If image is already in RGB8 pixel format, the function
 * returns successfully.
 *
 * @param image Pointer to the `CIPR_Image` structure.
 * @return 0 on success, -1 on failure.
 */
int cipr_format_convert_to_RGB8(CIPR_Image *image);

/**
 * @brief Applies a Gaussian blur filter on a `CIPR_Image` structure.
 *
 * Performs a Gaussian blur on an image structure in-place using a separable
 * horizontal and vertical convolution schema. Each separable pass is optimized
 * with AVX2 vector intrinsics and is parallelized using the global thread
 * pool. The strength of the blur is controlled by `standard_deviation`.
 *
 * @param image Pointer to the `CIPR_Image` structure.
 * @param standard_deviation Standard deviation of the Gaussian kernel (> 0).
 * @return 0 on success, -1 on failure.
 */
int cipr_filter_blur_gaussian(CIPR_Image *image, float standard_deviation);

/**
 * @brief Applies a box blur filter on a `CIPR_Image` structure.
 *
 * Performs a box blur on an image structure in-place using a separable
 * horizontal and vertical convolution schema. Each separable pass is optimized
 * using AVX2 vector intrinsics and is parallelized using the global thread
 * pool. The strength of the blur is controlled by the desired kernel size,
 * `size`.
 *
 * @param image Pointer to the `CIPR_Image` structure.
 * @param size Size of the box blur kernel (odd, >= 3)
 * @return 0 on success, -1 on failure.
 */
int cipr_filter_blur_box(CIPR_Image *image, int size);

/**
 * @brief Applies an unsharp mask filter on a `CIPR_Image` structure.
 *
 * Performs sharpening on an image structure in-place using the unsharp masking
 * technique, which combines the original image with a blurred version.
 * The blurred image is obtained internally using the Gaussian blur filter with
 * the specified `standard_deviation`.
 *
 * @param image Pointer to the `CIPR_Image` structure (GRAY8).
 * @param amount Multiplier for the sharpening strength (> 0)
 * @param standard_deviation Standard deviation of the Gaussian blur (> 0).
 * @return 0 on success, -1 on failure.
 */
int cipr_filter_unsharp_mask(CIPR_Image *image, float amount, float standard_deviation);

/**
 * @brief Applies a Sobel operator filter on a `CIPR_Image` structure.
 *
 * Performs a Sobel operator filter on an image structure in-place using a
 * separable horizontal and vertical convolution schema. Each separable pass is
 * optimized with AVX2 vector intrinsics and is parallelized using the global
 * thread pool. The image structure must be in GRAY8 pixel format prior to
 * calling this function.
 *
 * @param image Pointer to the `CIPR_Image` structure (GRAY8).
 * @return 0 on success, -1 on failure.
 */
int cipr_filter_sobel_operator(CIPR_Image *image);

// ----------------------------------------------------------------------------
// Legacy Gaussian blur and box blur algorithms
// ----------------------------------------------------------------------------

/**
 * Note: these algorithms are only included for benchmarking and future
 * reference. They do not make use of the global thread pool.
 */

// Enumeration of legacy Gaussian blur implementations.
typedef enum CIPR_GAUSSIAN_IMPL {
    GAUSSIAN_NAIVE = 0,
    GAUSSIAN_SEPARABLE,
    GAUSSIAN_SEPARABLE_AVX2,
} CIPR_GAUSSIAN_IMPL;

// Performs a Gaussian blur filter using a selected legacy implementation
int cipr_legacy_filter_blur_gaussian(CIPR_Image *image, float standard_deviation,
                                     CIPR_GAUSSIAN_IMPL implementation);

// Enumeration of legacy box blur implementations.
typedef enum CIPR_BLUR_BOX_IMPL {
    BLUR_BOX_NAIVE = 0,
    BLUR_BOX_SEPARABLE,
    BLUR_BOX_RUNNING_SUM,
    BLUR_BOX_RUNNING_SUM_TRANSPOSE,
    BLUR_BOX_SEPARABLE_AVX2,
} CIPR_BLUR_BOX_IMPL;

// Performs a box blur filter using a selected legacy implementation
int cipr_legacy_filter_blur_box(CIPR_Image *image, int size, CIPR_BLUR_BOX_IMPL implementation);

#ifdef __cplusplus
}
#endif

#endif // LIBCIPR_LIBCIPR_H_