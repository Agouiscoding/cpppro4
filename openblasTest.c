#include <cblas.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
struct timeval start, end;
long timeuse;

void main()
{

  size_t n = 10000;
  static double a[100000000];
  static double b[100000000];
  // static double c[100000000];
  printf("For the %ld * %ld matrix:\n", n, n);


  for (size_t i = 0; i < n * n; i++)
  {
    a[i] = rand();
    b[i] = rand();
  }

  gettimeofday(&start, NULL);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, a, n, b, n, 0, a, n);
 
  gettimeofday(&end, NULL);

 timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
  printf("The Openblas method took %.3f ms\n", timeuse / 1000.0);


}