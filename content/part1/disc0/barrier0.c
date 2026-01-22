#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>

struct ThreadInfo {
  long *array;
  long  n;
  long  chunk;
  long  id;
  pthread_barrier_t *bar;
  pthread_t tid;
};
  
void * func(void *arg)
{
  struct ThreadInfo *ta = arg;
  long  id    = ta->id;
  long  chunk = ta->chunk;
  long *array = &(ta->array[id * chunk]);
  unsigned int seed;

  // phase 1: start
  printf("Hello from func %ld ... zeroing\n", id);
  for (long i=0; i<chunk; i++) array[i] = 0;  
  pthread_barrier_wait(ta->bar); // end phase 1

  // phase 2: start -- nothing to do
  pthread_barrier_wait(ta->bar); // end phase 2

  // phase 3: start 
  printf("           func %ld ... randomize\n", id);
  for (long i=0; i<chunk; i++) array[i] = rand_r(&seed);
  pthread_barrier_wait(ta->bar); // end phase 3

  // phase 4 start -- main is checking array while we print and exit
  printf("           func %ld ... all done\n", id);
  return NULL;
}

int main(int argc, char **argv)
{
  pthread_t tid;
  int rc, i;
  long n = 10;
  long chunk = 1000;
  long *array;
  struct ThreadInfo *tinfo;
  pthread_barrier_t bar;
  
  printf("Hello from main thread\n");
  
  if (argc >= 2) n = atoi(argv[1]);
  if (argc == 3) chunk = atoi(argv[2]);
  
  array = malloc(sizeof(long) * chunk * n);
  tinfo = malloc(sizeof(struct ThreadInfo) * n);
  
  rc = pthread_barrier_init(&bar, NULL, n+1);  
  assert(rc == 0);

  for (i=0; i<n; i++) {
    tinfo[i].array = array;
    tinfo[i].n     = n;
    tinfo[i].chunk = chunk;
    tinfo[i].id    = i;
    tinfo[i].bar   = &bar;
    rc = pthread_create(&(tinfo[i].tid), NULL, func,&(tinfo[i]));
    assert(rc == 0);
  }

  // phase 1 start -- main nothing to do
  pthread_barrier_wait(&bar);  // end phase 1

  // phase 2 start 
  for (long j=0; j< (chunk * n); j++) { assert(array[j] == 0); }
  pthread_barrier_wait(&bar); // end phase 2

  // phase 3 start -- main nothing to do
  pthread_barrier_wait(&bar); // end phase 3

  // phase 4 start
  for (long j=0; j< (chunk * n); j++) { assert(array[j] != 0); }

  for (i=0; i<n; i++) {   
    rc = pthread_join(tinfo[i].tid, NULL);
    assert(rc == 0);
  }
  // phase 5 -- main only
  free(array);
  free(tinfo);
  return 0;
}

