#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>

#define DIM 1024
long A[DIM], B[DIM], C[DIM];

// The work
void vecAdd(long *a, long *b, long *c, int dim) {
  for (int i=0; i<dim; i++) {
    c[i] = a[i] + b[i];
  }
}

// dummy load
void loadVec(long *v, int dim) {
  for (int i=0; i<dim; i++) {
    v[i] = rand();
  }
}

// struct to pass args to work thread
struct wargs { int dim; long *a, *b, *c; };

// entry point for additional worker thread
//  1. unpack arguments, and
//  2. call vecAdd
void *worker(void *arg) {
  struct wargs *wa = arg;
  vecAdd(wa->a, wa->b, wa->c, wa->dim);
}

int main(int argc, char **argv) {
  struct wargs wa;
  pthread_t tid;
  int rc, hd;

  // load A and B vectors
  loadVec(A, DIM);
  loadVec(B, DIM);

  // split work in half
  hd = DIM/2;
  
  // pack arguments for worker and create thread
  wa.a = A; wa.b = B; wa.c = C; wa.dim = hd;
  rc = pthread_create(&tid, NULL, worker, &wa);
  assert(rc == 0);

  // main thread takes care of remainder
  vecAdd(&A[hd], &B[hd], &C[hd], DIM-hd);

  // wait for worker
  rc = pthread_join(tid, NULL);
  assert(rc == 0);

  // Normally we would do something with C
  return 0;
}
