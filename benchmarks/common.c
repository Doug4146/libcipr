/**
 * @file common.c
 *
 * Implements the support functionality for benchmarking declared in
 * common.h.
 */

#include "common.h"
#include <stdint.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else // POSIX
#include <time.h>
#endif

void print_char_line(char character, int num)
{
    for (int i = 0; i < num; i++) {
        printf("%c", character);
    }
    printf("\n");
}

double array_find_minimum(double *array, int array_length)
{
    double minimum = array[0];
    for (int i = 0; i < array_length; i++) {
        if (array[i] < minimum) {
            minimum = array[i];
        }
    }
    return minimum;
}

uint64_t get_time_nanoseconds(void)
{
#ifdef _WIN32
    LARGE_INTEGER timer, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&timer);
    return (uint64_t)(((double)timer.QuadPart * 1000000000.0f) / (double)frequency.QuadPart);
#else // POSIX
    struct timespec timer;
    clock_gettime(CLOCK_MONOTONIC, &timer);
    return (uint64_t)((double)timer.tv_sec * 1000000000.0f) + (uint64_t)(timer.tv_nsec);
#endif
}