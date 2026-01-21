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
  pthread_attr_t attr;
  pthread_attr_t *attrp;
  pthread_t tid;
  int rc;
  
  printf("Hello from main thread\n");
  
  attrp = &attr;
  rc = pthread_attr_init(attrp);
  assert(rc == 0);
  
  rc = pthread_create(&tid, attrp, func, NULL);
  assert(rc == 0);
  
  return 0;
}
