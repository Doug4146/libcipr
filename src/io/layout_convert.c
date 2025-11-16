/**
 * @file layout_convert.c
 *
 * Implements the pixel layout conversion (including stride-based padding)
 * operations declared in io/layout_convert.h.
 *
 * Note: these operations could potentially be further optimized with SIMD
 * and/or multithreading.
 */

#include "io/layout_convert.h"
#include "utils/utils.h"

void cipr__layout_interleaved_to_planar_RGB8(cipr_u8 *dst_buffer, cipr_u8 *src_buffer, cipr_i32 h,
                                             cipr_i32 w, cipr_i32 stride)
{
    for (cipr_i32 y = 0; y < h; y++) {
        for (cipr_i32 x = 0; x < w; x++) {

            // Index of channels in non-padded interleaved layout
            cipr_i32 src_index_R = ((y * w + x) * 3) + 0;
            cipr_i32 src_index_G = ((y * w + x) * 3) + 1;
            cipr_i32 src_index_B = ((y * w + x) * 3) + 2;

            // index of channel in padded planar layout
            cipr_i32 dst_index_R = (0 * h * stride) + (y * stride + x);
            cipr_i32 dst_index_G = (1 * h * stride) + (y * stride + x);
            cipr_i32 dst_index_B = (2 * h * stride) + (y * stride + x);

            dst_buffer[dst_index_R] = src_buffer[src_index_R];
            dst_buffer[dst_index_G] = src_buffer[src_index_G];
            dst_buffer[dst_index_B] = src_buffer[src_index_B];
        }
    }
}

void cipr__layout_planar_to_interleaved_RGB8(cipr_u8 *dst_buffer, cipr_u8 *src_buffer, cipr_i32 h,
                                             cipr_i32 w, cipr_i32 stride)
{
    for (cipr_i32 y = 0; y < h; y++) {
        for (cipr_i32 x = 0; x < w; x++) {

            // Index of channels in padded planar layout
            cipr_i32 src_index_R = (0 * h * stride) + (y * stride + x);
            cipr_i32 src_index_G = (1 * h * stride) + (y * stride + x);
            cipr_i32 src_index_B = (2 * h * stride) + (y * stride + x);

            // Index of channels in non-padded interleaved layout
            cipr_i32 dst_index_R = ((y * w + x) * 3) + 0;
            cipr_i32 dst_index_G = ((y * w + x) * 3) + 1;
            cipr_i32 dst_index_B = ((y * w + x) * 3) + 2;

            dst_buffer[dst_index_R] = src_buffer[src_index_R];
            dst_buffer[dst_index_G] = src_buffer[src_index_G];
            dst_buffer[dst_index_B] = src_buffer[src_index_B];
        }
    }
}

void cipr__layout_unpad_GRAY8(cipr_u8 *dst_buffer, cipr_u8 *src_buffer, cipr_i32 h, cipr_i32 w,
                              cipr_i32 stride)
{
    for (cipr_i32 y = 0; y < h; y++) {
        for (cipr_i32 x = 0; x < w; x++) {

            // Index of channel in padded layout
            cipr_i32 src_index = y * stride + x;

            // Index of channel in non-padded layout
            cipr_i32 dst_index = y * w + x;

            dst_buffer[dst_index] = src_buffer[src_index];
        }
    }
}