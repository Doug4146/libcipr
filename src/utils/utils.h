/**
 * @file utils.h
 *
 * Internal header for common utility functionality including common types,
 * aligned memory allocation and freeing, and small math helpers.
 */

#ifndef LIBCIPR_UTILS_UTILS_H_
#define LIBCIPR_UTILS_UTILS_H_

#include <stddef.h>
#include <stdint.h>

// Internal type typedefs
typedef uint8_t cipr_u8;
typedef uint16_t cipr_u16;
typedef int16_t cipr_i16;
typedef int32_t cipr_i32;
typedef uint32_t cipr_u32;
typedef long cipr_long;
typedef float cipr_f32;
typedef size_t cipr_usize;

// Pi definition
#define CIPR__PI 3.141592653589793

// Returns the greater of two cipr_u8 values, `a`, and `b`
static inline cipr_u8 cipr__max_pair_u8(cipr_u8 a, cipr_u8 b)
{
    return (a >= b) ? a : b;
}

// Returns the lower of two cipr_u8 values, `a` and `b`
static inline cipr_u8 cipr__min_pair_u8(cipr_u8 a, cipr_u8 b)
{
    return (a <= b) ? a : b;
}

// Returns the value of an cipr_i32, `a`, clamped to the range of cipr_u8
static inline cipr_u8 cipr__clamp_u8_i32(cipr_i32 a)
{
    cipr_u8 out_value = a <= 0 ? 0 : (a >= 255 ? 255 : a);
    return out_value;
}

// Returns the value of a cipr_f32, `a`, clamped to the range of cipr_u8
static inline cipr_u8 cipr__clamp_u8_f32(cipr_f32 a)
{
    cipr_u8 out_value = a <= 0 ? 0 : (a >= 255 ? 255 : a);
    return out_value;
}

// Returns the value of `size` rounded up to the nearest multiple of `value`
static inline cipr_usize cipr__round_up(cipr_usize size, cipr_usize value)
{
    cipr_usize remainder = size % value;
    cipr_usize out_value = remainder == 0 ? size : size + (value - remainder);
    return out_value;
}

/**
 * @brief Allocates `size` bytes aligned to `alignment` bytes.
 *
 * Internally uses `posix_memalign` on POSIX compatible systems, or
 * `_aligned_malloc` on Windows systems. Must free allocated memory with
 * `cipr__aligned_free`.
 *
 * @return Pointer to allocated memory, or NULL on failure.
 */
void *cipr__aligned_alloc(cipr_usize size, cipr_usize alignment);

/**
 * @brief Frees aligned allocated memory at address `ptr`.
 *
 * Internally, uses `free` on POSIX compatible systems, or `_aligned_free` on
 * Windows systems. Memory must have been allocated with `cipr__aligned_alloc`.
 */
void cipr__aligned_free(void *ptr);

#endif // LIBCIPR_UTILS_UTILS_H_