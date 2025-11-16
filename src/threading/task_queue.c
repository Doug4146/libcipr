/**
 * @file task_queue.c
 *
 * Implements the task queue structure operations declared in
 * threading/task_queue.h.
 */

#include "threading/task_queue.h"
#include "utils/utils.h"
#include <stdbool.h>
#include <stdlib.h>

struct CIPR__TaskQueue *cipr__task_queue_create(cipr_i32 capacity)
{
    // Allocate struct
    struct CIPR__TaskQueue *queue = malloc(sizeof(struct CIPR__TaskQueue));
    if (queue == NULL) {
        return NULL;
    }

    // Allocate tasks array
    queue->tasks = malloc(capacity * sizeof(struct CIPR__Task));
    if (queue->tasks == NULL) {
        free(queue);
        return NULL;
    }

    // Initialize fields
    queue->capacity = capacity;
    queue->num_tasks = 0;
    queue->front = 0;
    queue->rear = 0;

    return queue;
}

void cipr__task_queue_destroy(struct CIPR__TaskQueue **queue)
{
    // Free and nullify tasks array field
    if ((*queue)->tasks != NULL) {
        free((*queue)->tasks);
        (*queue)->tasks = NULL;
    }

    // Free and nullify struct
    if (*queue != NULL) {
        free(*queue);
        *queue = NULL;
    }
}

bool cipr__task_queue_full(struct CIPR__TaskQueue *queue)
{
    return (queue->num_tasks == (cipr_i32)queue->capacity);
}

bool cipr__task_queue_empty(struct CIPR__TaskQueue *queue)
{
    return (queue->num_tasks == 0);
}

int cipr__task_queue_enqueue(struct CIPR__TaskQueue *queue, struct CIPR__Task task)
{
    // Check if full
    if (cipr__task_queue_full(queue)) {
        return -1;
    }

    // Add task to rear and update queue variables
    queue->tasks[queue->rear] = task;
    queue->rear = (queue->rear + 1) % queue->capacity;
    queue->num_tasks++;

    return 0;
}

int cipr__task_queue_dequeue(struct CIPR__TaskQueue *queue, struct CIPR__Task *task_out)
{
    // Check if empty
    if (cipr__task_queue_empty(queue)) {
        return -1;
    }

    // Update task_out with the task to dequeue and update queue variables
    *task_out = queue->tasks[queue->front];
    queue->front = (queue->front + 1) % queue->capacity;
    queue->num_tasks--;

    return 0;
}