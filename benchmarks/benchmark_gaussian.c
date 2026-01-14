#include "common.h"
#include <LIBCIPR/libcipr.h>
#include <stdint.h>
#include <stdio.h>

static int benchmark_legacy_function(CIPR_GAUSSIAN_IMPL implementation, double *times_array,
                                     double *min_time_ms);
static int benchmark_library_function(double *times_array, double *min_time_ms);

// Control the number of iterations done in benchmarking
#define MAX_ITERATIONS 5

// // Control the image file used for benchmarking
// #define INPUT_PATH (const char*) "../benchmarks/input_images/7680x4320.jpeg"
// #define INPUT_PATH (const char*) "../benchmarks/input_images/3840x2160.jpeg"
// #define INPUT_PATH (const char*) "../benchmarks/input_images/2560x1440.jpeg"
// #define INPUT_PATH (const char*) "../benchmarks/input_images/1920x1080.jpeg"
#define INPUT_PATH (const char *)"../benchmarks/input_images/1280x720.jpeg"

// Control the standard deviation of the Gaussian blur
#define STANDARD_DEVIATION 2.5

// Control the number of threads used for the final multithreaded function
#define NUM_THREADS 8

int main(void)
{

    if (cipr_thread_pool_init(NUM_THREADS) != 0) {
        return -1;
    }

    double times_array[MAX_ITERATIONS] = {0};

    // --------------------------------------------------------------------
    // Benchmarking for the legacy naive function
    // --------------------------------------------------------------------

    double min_time_naive_ms = 0;
    if (benchmark_legacy_function(GAUSSIAN_NAIVE, times_array, &min_time_naive_ms) != 0) {
        cipr_thread_pool_shutdown();
        return -1;
    }

    // --------------------------------------------------------------------
    // Benchmarking for the legacy separable function
    // --------------------------------------------------------------------

    double min_time_separable_ms = 0;
    if (benchmark_legacy_function(GAUSSIAN_SEPARABLE, times_array, &min_time_separable_ms) != 0) {
        cipr_thread_pool_shutdown();
        return -1;
    }

    // --------------------------------------------------------------------
    // Benchmarking for the legacy separable AVX2 function
    // --------------------------------------------------------------------

    double min_time_separable_avx2_ms = 0;
    if (benchmark_legacy_function(GAUSSIAN_SEPARABLE_AVX2, times_array,
                                  &min_time_separable_avx2_ms) != 0) {
        cipr_thread_pool_shutdown();
        return -1;
    }

    // --------------------------------------------------------------------
    // Benchmarking for the legacy separable AVX2 fixed-point function
    // --------------------------------------------------------------------

    double min_time_separable_avx2_fixed_point_ms = 0;
    if (benchmark_legacy_function(GAUSSIAN_SEPARABLE_AVX2_FIXED_POINT, times_array,
                                  &min_time_separable_avx2_fixed_point_ms) != 0) {
        cipr_thread_pool_shutdown();
        return -1;
    }

    // --------------------------------------------------------------------
    // Benchmarking for the final multithreaded (main library) function
    // --------------------------------------------------------------------

    // Note the final multithreaded function utilizes the separable AVX2 algorithm
    double min_time_final_mt_ms = 0;
    if (benchmark_library_function(times_array, &min_time_final_mt_ms) != 0) {
        cipr_thread_pool_shutdown();
        return -1;
    }

    // --------------------------------------------------------------------
    // Printing the results
    // --------------------------------------------------------------------

    printf("\n\n");
    print_char_line('-', 56);
    printf("  Benchmarking Gaussian Blur       |  Iterations = %d   \n", MAX_ITERATIONS);
    print_char_line('-', 56);
    printf("\n");

    printf("     %-29s  |  %17s \n", "Function", "Minimum Time (ms)");
    print_char_line('-', 56);
    printf("     %-29s  |  %9.3f \n", "Naive", min_time_naive_ms);
    printf("     %-29s  |  %9.3f \n", "Separable", min_time_separable_ms);
    printf("     %-29s  |  %9.3f \n", "Separable (AVX2)", min_time_separable_avx2_ms);
    printf("     %-29s  |  %9.3f \n", "Separable (AVX2, fixed-point)", min_time_separable_avx2_fixed_point_ms);
    printf("     %-29s  |  %9.3f \n", "Final (Multithreaded)", min_time_final_mt_ms);
    printf("\n\n");

    cipr_thread_pool_shutdown();

    return 0;
}

static int benchmark_legacy_function(CIPR_GAUSSIAN_IMPL implementation, double *times_array,
                                     double *min_time_ms)
{
    for (int i = 0; i < MAX_ITERATIONS; i++) {

        CIPR_Image *image = cipr_image_create();
        if (image == NULL) {
            return -1;
        }

        if (cipr_io_read_jpeg(image, INPUT_PATH) != 0) {
            cipr_image_destroy(&image);
            return -1;
        }

        uint64_t start_time = get_time_nanoseconds();

        // Apply the specified Gaussian blur function
        if (cipr_legacy_filter_blur_gaussian(image, STANDARD_DEVIATION, implementation) != 0) {
            cipr_image_destroy(&image);
            return -1;
        }

        uint64_t end_time = get_time_nanoseconds();
        times_array[i] = (double)(end_time - start_time) / 1000000.0f; // milliseconds

        cipr_image_destroy(&image);
    }

    *min_time_ms = array_find_minimum(times_array, MAX_ITERATIONS);

    return 0;
}

static int benchmark_library_function(double *times_array, double *min_time_ms)
{
    for (int i = 0; i < MAX_ITERATIONS; i++) {

        CIPR_Image *image = cipr_image_create();
        if (image == NULL) {
            return -1;
        }

        if (cipr_io_read_jpeg(image, INPUT_PATH) != 0) {
            cipr_image_destroy(&image);
            return -1;
        }

        uint64_t start_time = get_time_nanoseconds();

        // Apply the final (multithreaded) Gaussian blur function
        if (cipr_filter_blur_gaussian(image, STANDARD_DEVIATION) != 0) {
            cipr_image_destroy(&image);
            return -1;
        }

        uint64_t end_time = get_time_nanoseconds();
        times_array[i] = (double)(end_time - start_time) / 1000000.0f; // milliseconds

        cipr_image_destroy(&image);
    }

    *min_time_ms = array_find_minimum(times_array, MAX_ITERATIONS);

    return 0;
}