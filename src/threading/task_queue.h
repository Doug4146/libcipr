/**
 * @file task_queue.h
 *
 * Internal header for the task structure, task queue structure, and its related
 * operations.
 */

#ifndef LIBCIPR_THREADING_TASK_QUEUE_H_
#define LIBCIPR_THREADING_TASK_QUEUE_H_

#include "utils/utils.h"
#include <stdbool.h>

// Definition of the task structure
struct CIPR__Task {
    void *(*func)(void *);
    void *arg;
};

// Definition of an array-based task queue structure
struct CIPR__TaskQueue {
    cipr_usize capacity;
    struct CIPR__Task *tasks;
    cipr_i32 num_tasks;
    cipr_i32 front;
    cipr_i32 rear;
};

/**
 * @brief Creates a `CIPR__TaskQueue` structure.
 *
 * Allocates memory for a task queue structure and initializes all fields. The
 * tasks array is alloacted enough space to store `capacity` number of tasks.
 *
 * @return Pointer to the `CIPR__TaskQueue` structure, or NULL upon failure.
 */
struct CIPR__TaskQueue *cipr__task_queue_create(cipr_i32 capacity);

/**
 * @brief Destroys a `CIPR__TaskQueue` structure.
 *
 * Frees all memory allocated for a task queue structure, including the tasks
 * array field.
 */
void cipr__task_queue_destroy(struct CIPR__TaskQueue **queue);

// Returns true if the `CIPR__TaskQueue` is full, false otherwise
bool cipr__task_queue_full(struct CIPR__TaskQueue *queue);

// Returns true if the `CIPR__TaskQueue` is empty, false otherwise
bool cipr__task_queue_empty(struct CIPR__TaskQueue *queue);

/**
 * @brief Enqueues a `CIPR__TaskQueue` structure with a `CIPR__Task` structure.
 *
 * Adds a task structure to a task queue structure. The queue behaves as a
 * circular buffer and the operation fails if the queue is full.
 *
 * @return 0 on success, -1 on failure.
 */
int cipr__task_queue_enqueue(struct CIPR__TaskQueue *queue, struct CIPR__Task task);

/**
 * @brief Dequeues a `CIPR__TaskQueue`.
 *
 * Removes a task structure from a task queue structure and writes the dequeued
 * task's fields to the specified `task_out` structure. The operation fails if
 * the queue is empty.
 *
 * @param task_out Pointer to `CIPR__Task` structure to write to.
 * @return 0 on success, -1 on failure.
 */
int cipr__task_queue_dequeue(struct CIPR__TaskQueue *queue, struct CIPR__Task *task_out);

#endif // LIBCIPR_THREADING_TASK_QUEUE_H_