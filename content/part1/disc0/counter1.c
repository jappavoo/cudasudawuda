#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>

pthread_barrier_t bar;
int counter = 0;

void *func(void *)
{
  pthread_barrier_wait(&bar);
  for (int i=0; i<1000; i++) counter++;
}

int main(int argc, char **argv)
{
  pthread_t *tid;
  int rc, i, n=10;
  
  if (argc == 2) n = atoi(argv[1]);
  tid = malloc(sizeof(pthread_t) * n);

  rc = pthread_barrier_init(&bar, NULL, n);  
  assert(rc == 0);

  for (i=0; i<n; i++) {
    rc = pthread_create(&(tid[i]), NULL, func, NULL);
    assert(rc == 0);
  }

  for (i=0; i<n; i++) {
    rc = pthread_join(tid[i], NULL);
    assert(rc == 0);
  }

  free(tid);
  printf("main thread counter=%d\n", counter);
  return 0;
}


