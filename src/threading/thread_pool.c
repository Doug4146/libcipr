/**
 * @file task_queue.c
 *
 * Implements the thread pool structure and its operations declared in
 * threading/thread_pool and LIBCIPR/libcipr. Contains the global thread
 * pool instance.
 */

#include "threading/thread_pool.h"
#include "LIBCIPR/libcipr.h"
#include "threading/task_queue.h"
#include "utils/utils.h"
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>

// Definition of the thread pool structure
struct CIPR__ThreadPool {
    bool stop;
    pthread_mutex_t mutex;
    pthread_cond_t signal;
    pthread_t *threads;
    cipr_usize num_threads;
    struct CIPR__TaskQueue *queue;
    cipr_i32 num_active_tasks;
    pthread_cond_t tasks_done;
};

// Global thread pool instance
static struct CIPR__ThreadPool g_tpool;

// Global variable for checking if global thread pool is initialized
static bool g_tpool_init = false;

bool cipr__thread_pool_is_init(void)
{
    return g_tpool_init;
}

int cipr__thread_pool_num_threads(void)
{
    return g_tpool.num_threads;
}

// Definition of the function each thread will execute throughout its lifetime
static void *thread_worker_function(void *arg)
{
    arg = NULL; // Not used

    while (1) {

        // Lock the mutex variable
        pthread_mutex_lock(&g_tpool.mutex);

        // Wait for the condition variable signal while the task queue is empty
        while (cipr__task_queue_empty(g_tpool.queue) && !g_tpool.stop) {
            pthread_cond_wait(&g_tpool.signal, &g_tpool.mutex);
        }

        // Unlock the mutex and exit the loop if stop is true
        if (g_tpool.stop) {
            pthread_mutex_unlock(&g_tpool.mutex);
            break;
        }

        // Perform dequeue operation and retrieve the task struct
        struct CIPR__Task task;
        cipr__task_queue_dequeue(g_tpool.queue, &task);

        // Unlock the mutex variable
        pthread_mutex_unlock(&g_tpool.mutex);

        // Call function specified by the task
        task.func(task.arg);

        // Lock the mutex
        pthread_mutex_lock(&g_tpool.mutex);

        // If all tasks done, signal the tasks_done condition variable
        g_tpool.num_active_tasks--;
        if ((g_tpool.num_active_tasks == 0) && cipr__task_queue_empty(g_tpool.queue)) {
            pthread_cond_signal(&g_tpool.tasks_done);
        }

        // Unlock the mutex
        pthread_mutex_unlock(&g_tpool.mutex);
    }

    return NULL;
}

int cipr_thread_pool_init(int num_threads)
{
    // Validate parameter
    if (num_threads < 1) {
        return -1;
    }

    // Check if the global thread pool is already initialized
    if (cipr__thread_pool_is_init()) {
        return -1;
    }

    // Allocate thread array
    g_tpool.threads = malloc(num_threads * sizeof(pthread_t));
    if (g_tpool.threads == NULL) {
        return -1;
    }

    // Create a task queue struct
    cipr_usize max_num_tasks = num_threads * 4;
    g_tpool.queue = cipr__task_queue_create(max_num_tasks);
    if (g_tpool.queue == NULL) {
        free(g_tpool.threads);
        return -1;
    }

    // Initialize the thread pool struct fields
    g_tpool.stop = false;
    pthread_mutex_init(&g_tpool.mutex, NULL);
    pthread_cond_init(&g_tpool.signal, NULL);
    g_tpool.num_threads = num_threads;
    g_tpool.num_active_tasks = 0;
    pthread_cond_init(&g_tpool.tasks_done, NULL);

    // Create the threads and pass the thread worker function
    for (cipr_i32 n = 0; n < num_threads; n++) {
        pthread_create(&g_tpool.threads[n], NULL, thread_worker_function, NULL);
    }

    // Set the global initialization variable
    g_tpool_init = true;

    return 0;
}

void cipr_thread_pool_shutdown(void)
{
    // Check if the global thread pool is not initialized
    if (!cipr__thread_pool_is_init()) {
        return;
    }

    // Lock the mutex
    pthread_mutex_lock(&g_tpool.mutex);

    // Set the stop field to true and broadcast the signal to all threads
    g_tpool.stop = true;
    pthread_cond_broadcast(&g_tpool.signal);

    // Unlock the mutex
    pthread_mutex_unlock(&g_tpool.mutex);

    // Wait for all the threads to join
    for (cipr_i32 n = 0; n < (cipr_i32)g_tpool.num_threads; n++) {
        pthread_join(g_tpool.threads[n], NULL);
    }

    // Destroy the mutex and condition variables
    pthread_mutex_destroy(&g_tpool.mutex);
    pthread_cond_destroy(&g_tpool.signal);
    pthread_cond_destroy(&g_tpool.tasks_done);

    // Free and nullify the threads array
    free(g_tpool.threads);
    g_tpool.threads = NULL;

    // Destroy and nullify the task queue struct
    cipr__task_queue_destroy(&g_tpool.queue);

    // Set the global initialization variable
    g_tpool_init = false;
}

int cipr__thread_pool_submit(void *(*func)(void *), void *arg)
{
    // Create a task struct
    struct CIPR__Task task = {.func = func, .arg = arg};

    // Lock the mutex
    pthread_mutex_lock(&g_tpool.mutex);

    // Add the task to the queue, on failure unlock the mutex and return
    if (cipr__task_queue_enqueue(g_tpool.queue, task) != 0) {
        pthread_mutex_unlock(&g_tpool.mutex);
        return -1;
    }
    g_tpool.num_active_tasks++;

    // Signal to a thread that a task is available in the queue
    pthread_cond_signal(&g_tpool.signal);

    // Unlock the mutex
    pthread_mutex_unlock(&g_tpool.mutex);

    return 0;
}

void cipr__thread_pool_wait_all(void)
{
    // Lock the mutex
    pthread_mutex_lock(&g_tpool.mutex);

    // Wait on the tasks_done condition variable while there are still active tasks
    while (g_tpool.num_active_tasks > 0) {
        pthread_cond_wait(&g_tpool.tasks_done, &g_tpool.mutex);
    }

    // Unlock the mutex
    pthread_mutex_unlock(&g_tpool.mutex);
}