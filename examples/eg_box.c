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

    // Apply the box blur filter to the image with size=15
    printf("Applying box blur...\n");
    if (cipr_filter_blur_box(image, 15) != 0) {
        cipr_image_destroy(&image);
        cipr_thread_pool_shutdown();
        fprintf(stderr, "Error: failed to apply box blur.\n");
        return -1;
    }
    printf("Box blur applied.\n");

    // Write the image structure to a JPEG file with quality=95
    printf("Writing image file...\n");
    if (cipr_io_write_jpeg(image, "../examples/output_images/box.jpeg", 95) != 0) {
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