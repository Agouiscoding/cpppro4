#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#pragma GCC optimize(3)
#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math")
#include "matrixmul.h"

struct timeval start, end;
long timeuse;

int main()
{
    size_t n = 10000;
    static float a[100000000];
    static float b[100000000];
    printf("For the %ld * %ld matrix:\n", n, n);

    for (size_t i = 0; i < n * n; i++)
    {
        a[i] = rand();
        b[i] = rand();
    }

    Matrix *m1 = createMatrix_initial(m1, n, n, a);
    Matrix *m2 = createMatrix_initial(m2, n, n, b);


    gettimeofday(&start, NULL);
    Matrix *product1 = matmul_plain(m1, m2);
    gettimeofday(&end, NULL);
    timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("The plain method took %.3f ms\n", timeuse / 1000.0);
    // printf("m1*m2:\n");
    // printMatrix(product1);
    deleteMatrix(&product1);

    gettimeofday(&start, NULL);
    Matrix *product2 = matmul_improved1(m1, m2);
    gettimeofday(&end, NULL);
    timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("The Optimizing memory access method took %.3f ms\n", timeuse / 1000.0);
    // printf("m1*m2:\n");
    // printMatrix(product2);
    deleteMatrix(&product2);

    gettimeofday(&start, NULL);
    Matrix *product3 = matmul_improved2(m1, m2);
    gettimeofday(&end, NULL);
    timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("The Optimizing memory access and omp method took %.3f ms\n", timeuse / 1000.0);
    // printf("m1*m2:\n");
    // printMatrix(product3);
    deleteMatrix(&product3);

    gettimeofday(&start, NULL);
    Matrix *product4 = matmul_improved3(m1, m2);
    gettimeofday(&end, NULL);
    timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("The omp and avx method took %.3f ms\n", timeuse / 1000.0);
    // printf("m1*m2:\n");
    // printMatrix(product4);
    deleteMatrix(&product4);

    gettimeofday(&start, NULL);
    Matrix *product5 = matmul_improved4(m1, m2);
    gettimeofday(&end, NULL);
    timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("The block matrix and omp took %.3f ms\n", timeuse / 1000.0);
    // printf("m1*m2:\n");
    // printMatrix(product4);
    deleteMatrix(&product5);

    deleteMatrix(&m1);
    deleteMatrix(&m2);
    return 0;
}
