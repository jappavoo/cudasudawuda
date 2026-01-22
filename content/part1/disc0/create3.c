#include <stdio.h>
#include <assert.h>
#include <pthread.h>

void * func(void *arg)
{
  printf("Hello from func\n");
  return NULL;
}

int main()
{
  pthread_t tid[10];
  int rc, i;

  printf("Hello from main thread\n");

  for (i=0; i<10; i++) {
    rc = pthread_create(&(tid[i]), NULL, func, NULL);
    assert(rc == 0);
  }

  for (i=0; i<10; i++) {
    rc = pthread_join(tid[i], NULL);
    assert(rc == 0);
  }
  
  return 0;
}

