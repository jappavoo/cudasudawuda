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
  pthread_t tid;
  int rc;
  
  printf("Hello from main thread\n");
  
  rc = pthread_create(&tid, NULL, func, NULL);
  assert(rc == 0);

  rc = pthread_join(tid, NULL);
  assert(rc == 0);
  
  return 0;
}
