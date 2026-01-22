#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>

struct ThreadArgs {
  long in;
  long out;
  long id;
};

void * func(void *arg)
{
  struct ThreadArgs *targ = arg;
  long id = targ->id;
  printf("Hello from func %ld: %ld\n", id, targ->in);
  targ->out = targ->in + id;
  return (void *)id;
}

int main()
{
  pthread_t tid[10];
  struct ThreadArgs args[10];
  int rc;
  long i;

  printf("Hello from main thread\n");

  for (i=0; i<10; i++) {
    args[i].id = i;
    args[i].in = rand();
    rc = pthread_create(&(tid[i]), NULL, func, &(args[i]));
    assert(rc == 0);
  }

  for (i=0; i<10; i++) {
    void *ret;
    rc = pthread_join(tid[i], &ret);
    assert(rc == 0);
    assert((long) ret == i);
    printf("%ld produced: out=%ld\n", i, args[i].out);
  }
  
  return 0;
}

