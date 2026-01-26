#include <string.h>
#include <stdlib.h>

#define DIM 1024

long A[DIM], B[DIM], C[DIM];

void
vecAdd(long *a, long *b, long *c, int dim)
{
  for (int i=0; i<dim; i++) {
    c[i] = a[i] + b[i];
  }
}

void loadVec(long *v, int dim)
{
  for (int i=0; i<dim; i++) {
    v[i] = rand();
  }
}

int main(int argc, char **argv)
{
  loadVec(A, DIM);
  loadVec(B, DIM);
  bzero(C, sizeof(C));

  vecAdd(A, B, C, DIM);
  
  return 0;
}
