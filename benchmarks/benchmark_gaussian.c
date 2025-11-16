#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else // POSIX
#include <time.h>
#endif

#include <LIBCIPR/libcipr.h>

static double find_minimum(double *array, int array_length);
static int benchmark_function(int (*blur_function)(CIPR_Image *, float), double *times_array,
                              double *min_time_ms);
static void print_char_line(char c, int n);

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
    if (benchmark_function(cipr_legacy_blur_gaussian_naive, times_array, &min_time_naive_ms) != 0) {
        cipr_thread_pool_shutdown();
        return -1;
    }

    // --------------------------------------------------------------------
    // Benchmarking for the legacy separable function
    // --------------------------------------------------------------------

    double min_time_separable_ms = 0;
    if (benchmark_function(cipr_legacy_blur_gaussian_separable, times_array,
                           &min_time_separable_ms) != 0) {
        cipr_thread_pool_shutdown();
        return -1;
    }

    // --------------------------------------------------------------------
    // Benchmarking for the legacy separable AVX2 function
    // --------------------------------------------------------------------

    double min_time_separable_avx2_ms = 0;
    if (benchmark_function(cipr_legacy_blur_gaussian_separable_avx2, times_array,
                           &min_time_separable_avx2_ms) != 0) {
        cipr_thread_pool_shutdown();
        return -1;
    }

    // --------------------------------------------------------------------
    // Benchmarking for the final multithreaded (main library) function
    // --------------------------------------------------------------------

    // Note the final multithreaded function utilizes the separable AVX2 algorithm
    double min_time_final_mt_ms = 0;
    if (benchmark_function(cipr_filter_blur_gaussian, times_array, &min_time_final_mt_ms) != 0) {
        cipr_thread_pool_shutdown();
        return -1;
    }

    // --------------------------------------------------------------------
    // Printing the results
    // --------------------------------------------------------------------

    printf("\n\n");
    print_char_line('-', 51);
    printf("  Benchmarking Gaussian Blur  |  Iterations = %d   \n", MAX_ITERATIONS);
    print_char_line('-', 51);
    printf("\n");

    printf("     %-23s  |  %17s \n", "Function", "Minimum Time (ms)");
    print_char_line('-', 51);
    printf("     %-23s  |  %9.3f \n", "Naive", min_time_naive_ms);
    printf("     %-23s  |  %9.3f \n", "Separable", min_time_separable_ms);
    printf("     %-23s  |  %9.3f \n", "Separable (AVX2)", min_time_separable_avx2_ms);
    printf("     %-23s  |  %9.3f \n", "Final (Multithreaded)", min_time_final_mt_ms);
    printf("\n\n");

    cipr_thread_pool_shutdown();

    return 0;
}

static void print_char_line(char c, int n)
{
    for (int i = 0; i < n; i++) {
        printf("%c", c);
    }
    printf("\n");
}

static double find_minimum(double *array, int array_length)
{
    double minimum = array[0];
    for (int i = 1; i < array_length; i++) {
        if (array[i] < minimum) {
            minimum = array[i];
        }
    }
    return minimum;
}

static int benchmark_function(int (*blur_function)(CIPR_Image *, float), double *times_array,
                              double *min_time_ms)
{
    for (int n = 0; n < MAX_ITERATIONS; n++) {

        CIPR_Image *image = cipr_image_create();
        if (image == NULL) {
            cipr_thread_pool_shutdown();
            return -1;
        }

        if (cipr_io_read_jpeg(image, INPUT_PATH) != 0) {
            cipr_image_destroy(&image);
            cipr_thread_pool_shutdown();
            return -1;
        }

#ifdef _WIN32
        LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
        LARGE_INTEGER Frequency;
        QueryPerformanceFrequency(&Frequency);
        QueryPerformanceCounter(&StartingTime);
#else // POSIX
        struct timespec StartingTime, EndingTime;
        clock_gettime(CLOCK_MONOTONIC, &StartingTime);
#endif

        // Apply the specified Gaussian blur function
        if (blur_function(image, STANDARD_DEVIATION) != 0) {
            cipr_image_destroy(&image);
            cipr_thread_pool_shutdown();
            return -1;
        }

#ifdef _WIN32
        QueryPerformanceCounter(&EndingTime);
        ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
        ElapsedMicroseconds.QuadPart *= 1000000;
        ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
        times_array[n] = (double)ElapsedMicroseconds.QuadPart / 1000;
#else // POSIX
        clock_gettime(CLOCK_MONOTONIC, &EndingTime);
        times_array[n] = ((EndingTime.tv_sec - StartingTime.tv_sec) * 1000.0) +
                         ((EndingTime.tv_nsec - StartingTime.tv_nsec) / 1e6);
#endif

        cipr_image_destroy(&image);
    }

    *min_time_ms = find_minimum(times_array, MAX_ITERATIONS);

    return 0;
}