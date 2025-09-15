#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <time.h>

#define CLOCK_SOURCE CLOCK_MONOTONIC
#define NSEC_IN_SECOND (1000000000)

typedef struct timespec ts_t;

static inline int ts_now(ts_t *now) {
  if (clock_gettime(CLOCK_SOURCE, now) == -1) {
    perror("clock_gettime");
    assert(0);
    return 0;
  }
  return 1;
}

static inline uint64_t ts_diff(ts_t start, ts_t end)
{
  uint64_t diff =
    ((end.tv_sec - start.tv_sec) * NSEC_IN_SECOND) +
    (end.tv_nsec - start.tv_nsec);
  return diff;
} 


