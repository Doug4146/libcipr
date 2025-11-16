/**
 * @file utils.c
 *
 * Implements the aligned memory allocation and freeing functionality declared
 * in utils/utils.h.
 */

#include "utils/utils.h"
#include <stdlib.h> // posix_memalign/free
#ifdef _WIN32
#include <malloc.h> // _aligned_malloc/_aligned_free
#endif

void *cipr__aligned_alloc(cipr_usize size, cipr_usize alignment)
{
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else // POSIX
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

void cipr__aligned_free(void *ptr)
{
#ifdef _WIN32
    _aligned_free(ptr);
#else // POSIX
    free(ptr);
#endif
}