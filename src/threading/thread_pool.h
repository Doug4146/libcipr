/**
 * @file thread_pool.h
 *
 * Internal header for the operations for the global thread pool, including
 * checking for initialization, task submission, and waiting for all submitted
 * tasks to complete.
 */

#ifndef LIBCIPR_THREADING_THREAD_POOL_H_
#define LIBCIPR_THREADING_THREAD_POOL_H_

#include <stdbool.h>

// Returns true if global thread pool has been initialized, false otherwise
bool cipr__thread_pool_is_init(void);

/**
 * @brief Returns the number of threads in the global thread pool.
 *
 * This function assumes that the global thread pool has been initialized.
 *
 * @return Number of running threads.
 */
int cipr__thread_pool_num_threads(void);

/**
 * @brief Submits a task to the global thread pool.
 *
 * Submits a task specified by the function pointer `func` and the argument
 * pointer `arg` into the global thread pool instance. The operation fails if
 * the task capacity is reached.
 *
 * @return 0 on success, -1 on failure.
 */
int cipr__thread_pool_submit(void *(*func)(void *), void *arg);

/**
 * @brief Waits for all submitted tasks in the thread pool to complete.
 *
 * Blocks the calling thread until all submitted tasks in the thread pool have
 * finished.
 */
void cipr__thread_pool_wait_all(void);

#endif // LIBCIPR_THREADING_THREAD_POOL_H_