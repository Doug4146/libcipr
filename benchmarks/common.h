/**
 * @file common.h
 *
 * Internal header for common support functionality for benchmarking, including
 * array minimum value determination, cross-platform timing, and print helpers.
 */

#ifndef LIBCIPR_BENCHMARKS_COMMON_H_
#define LIBCIPR_BENCHMARKS_COMMON_H_

#include <stdint.h>

// Prints a character repeatedly, num times.
void print_char_line(char character, int num);

/**
 * @brief Determines the minimum value of an array of doubles.
 *
 * @param array Pointer to the array of doubles, assumed to be non-NULL.
 * @param array_length Length of the array, assumed to be at least than 1.
 * @return Minimum value of array as a double.
 */
double array_find_minimum(double *array, int array_length);

/**
 * @brief Returns the current monotonic time in milliseconds for cross-platform
 * benchmarking.
 *
 * @return Current time in milliseconds as a double.
 */
uint64_t get_time_nanoseconds(void);

#endif // LIBCIPR_BENCHMARKS_COMMON_H_