#include <LIBCIPR/libcipr.h>
#include <stdio.h>

int main(void)
{

    // Initialize the libcipr threadpool with 8 threads
    printf("Initializing libcipr...\n");
    if (cipr_thread_pool_init(8) != 0) {
        fprintf(stderr, "Error: failed to initialize libcipr.\n");
        return -1;
    }
    printf("libcipr initialized.\n");

    // Create an image structure
    printf("Creating CIPR_Image structure...\n");
    CIPR_Image *image = cipr_image_create();
    if (image == NULL) {
        cipr_thread_pool_shutdown();
        fprintf(stderr, "Error: failed to create CIPR_ structure.\n");
        return -1;
    }
    printf("CIPR_Image structure created.\n");

    // Read JPEG file into image structure
    printf("Reading image file...\n");
    if (cipr_io_read_jpeg(image, "../examples/input_images/1920x1282.jpeg") != 0) {
        cipr_image_destroy(&image);
        cipr_thread_pool_shutdown();
        fprintf(stderr, "Error: failed to read JPEG file.\n");
        return -1;
    }
    printf("Image file read.\n");

    // Convert the image's pixel format to GRAY8 using luminosity technique
    printf("Converting image to grayscale...\n");
    if (cipr_format_convert_to_GRAY8(image, CIPR_GRAY8_LUMINOSITY) != 0) {
        cipr_image_destroy(&image);
        cipr_thread_pool_shutdown();
        fprintf(stderr, "Error: failed to convert pixel format to GRAY8.\n");
    }
    printf("Image converted to grayscale.\n");

    // Apply the Gaussian blur filter to the image with stddev=1.25
    printf("Applying Gaussian blur...\n");
    if (cipr_filter_blur_gaussian(image, 1.25) != 0) {
        cipr_image_destroy(&image);
        cipr_thread_pool_shutdown();
        fprintf(stderr, "Error: failed to apply Gaussian blur.\n");
        return -1;
    }
    printf("Gaussian blur applied.\n");

    // Apply the Sobel operator to the image
    printf("Applying Sobel operator...\n");
    if (cipr_filter_sobel_operator(image) != 0) {
        cipr_image_destroy(&image);
        cipr_thread_pool_shutdown();
        fprintf(stderr, "Error: failed to apply Sobel operator.\n");
        return -1;
    }
    printf("Sobel operator applied.\n");

    // Convert the image's pixel format to RGB8 (will still "appear" gray)
    printf("Converting image to rgb...\n");
    if (cipr_format_convert_to_RGB8(image) != 0) {
        cipr_image_destroy(&image);
        cipr_thread_pool_shutdown();
        fprintf(stderr, "Error: failed to convert pixel format to RGB8.\n");
    }
    printf("Image converted to rgb.\n");

    // Apply the unsharp mask filter to the image with amount=2 and stddev=0.75
    printf("Applying unsharp mask...\n");
    if (cipr_filter_unsharp_mask(image, 2, 0.75) != 0) {
        cipr_image_destroy(&image);
        cipr_thread_pool_shutdown();
        fprintf(stderr, "Error: failed to apply unsharp mask.\n");
        return -1;
    }
    printf("Unsharp mask applied.\n");

    // Write the image structure to a JPEG file with quality=95
    printf("Writing image file...\n");
    if (cipr_io_write_jpeg(image, "../examples/output_images/combination.jpeg", 95) != 0) {
        cipr_image_destroy(&image);
        cipr_thread_pool_shutdown();
        fprintf(stderr, "Error: failed to write to a JPEG file.\n");
        return -1;
    }
    printf("Image file written.\n");

    // Destroy the image structure and shutdown the threadpool
    printf("Destroying CIPR_Image structure and shutting down libcipr...\n");
    cipr_image_destroy(&image);
    cipr_thread_pool_shutdown();
    printf("CIPR_Image structure destroyed and libcipr shutdown.\n");

    return 0;
}