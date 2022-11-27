#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include "matrixmul.h"
#pragma GCC optimize(3)
#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math")

Matrix *createMatrix_default(Matrix *matrix, size_t row, size_t col);               //声明一个m*n的矩阵并初始化为0
Matrix *createMatrix_initial(Matrix *matrix, size_t row, size_t col, float *array); //声明一个矩阵并用数组初始化
void valueMatrix_array(Matrix *matrix, float *array);                               //用数组给矩阵赋值
void deleteMatrix(Matrix **matrix);                                                 //删除一个矩阵 释放空间 指针置NULL
size_t sizeMatrix(Matrix *matrix);                                                  //返回矩阵的大小row*col
Matrix *matmul_plain(Matrix *matrix_left, Matrix *matrix_right);                    //朴素的矩阵相乘
Matrix *matmul_improved1(Matrix *matrix_left, Matrix *matrix_right);                //访存优化
Matrix *matmul_improved2(Matrix *matrix_left, Matrix *matrix_right);                //访存优化+openMP优化
Matrix *matmul_improved3(Matrix *matrix_left, Matrix *matrix_right);                // openMP+avx指令优化
Matrix *matmul_improved4(Matrix *matrix_left, Matrix *matrix_right);                // 矩阵分块
void printMatrix(Matrix *matrix);                                                   //打印矩阵

//声明一个m*n的矩阵并初始化为0
Matrix *createMatrix_default(Matrix *matrix, size_t row, size_t col)
{
    if (row > 0 && col > 0)
    {
        matrix = (Matrix *)malloc(sizeof(Matrix));
        matrix->row = row;
        matrix->col = col;
        matrix->data = (float *)malloc(sizeof(float) * row * col);
        memset(matrix->data, 0, sizeof(float) * row * col);
        return matrix;
    }
    else
    {
        printf("Error in createMatrix_default:The value of row or col must be greater than zero!\n");
        return NULL;
    }
}

//声明一个矩阵并用数组初始化
Matrix *createMatrix_initial(Matrix *matrix, size_t row, size_t col, float *array)
{
    if (row > 0 && col > 0)
    {
        matrix = (Matrix *)malloc(sizeof(Matrix));
        matrix->row = row;
        matrix->col = col;
        matrix->data = (float *)malloc(sizeof(float) * row * col);
        memcpy(matrix->data, array, matrix->row * matrix->col * sizeof(float));
        return matrix;
    }
    else
    {
        printf("Error in createMatrix_initial:The value of row or col must be greater than zero!\n");
        return NULL;
    }
}

//用数组给矩阵赋值
void valueMatrix_array(Matrix *matrix, float *array)
{
    if (matrix == NULL)
    {
        printf("Error in valueMatrix_array:The matrix does not exist or has been deleted!\n");
    }
    else
    {
        memcpy(matrix->data, array, matrix->row * matrix->col * sizeof(float));
    }
}

//删除一个矩阵 释放空间 指针置NULL
void deleteMatrix(Matrix **matrix)
{
    if (*matrix == NULL)
    {
        printf("Error in deleteMatrix:This matrix has already been deleted!\n");
        return;
    }
    free((*matrix)->data);
    (*matrix)->data = NULL;
    free(*matrix);
    *matrix = NULL;
}

//返回矩阵的大小row*col
size_t sizeMatrix(Matrix *matrix)
{
    return matrix->row * matrix->col;
}

//矩阵相乘
Matrix *matmul_plain(Matrix *matrix_left, Matrix *matrix_right)
{
    if (matrix_left == NULL || matrix_right == NULL)
    {
        printf("Error in mulMatrix:one of the matrix does not exist or has been deleted!\n");
        return NULL;
    }
    else if (matrix_left->col == matrix_right->row)
    {
        Matrix *matrix_product = createMatrix_default(matrix_product, matrix_left->row, matrix_right->col);
        size_t lrow = matrix_left->row;
        size_t lcol = matrix_left->col;
        size_t rcol = matrix_right->col;
        for (size_t i = 0; i < lrow; i++)
        {
            for (size_t j = 0; j < rcol; j++)
            {
                for (size_t k = 0; k < lcol; k++)
                {
                    matrix_product->data[i * matrix_product->col + j] += matrix_left->data[i * lcol + k] * matrix_right->data[k * rcol + j];
                }
            }
        }
        return matrix_product;
    }
    else
    {
        printf("Error in mulMatrix:matrix_left's column must equal to matrix_right's row!\n");
        return NULL;
    }
}

//访存优化
Matrix *matmul_improved1(Matrix *matrix_left, Matrix *matrix_right)
{
    if (matrix_left == NULL || matrix_right == NULL)
    {
        printf("Error in mulMatrix:one of the matrix does not exist or has been deleted!\n");
        return NULL;
    }
    else if (matrix_left->col == matrix_right->row)
    {
        Matrix *matrix_product = createMatrix_default(matrix_product, matrix_left->row, matrix_right->col);
        float r;
        size_t lrow = matrix_left->row;
        size_t lcol = matrix_left->col;
        size_t rcol = matrix_right->col;
        size_t rrow = matrix_right->row;

        for (size_t i = 0; i < lrow; i++)
        {
            for (size_t k = 0; k < rrow; k++)
            {
                r = matrix_left->data[i * lcol + k];
                for (size_t j = 0; j < rcol; j++)
                {
                    matrix_product->data[i * matrix_product->col + j] += (r * matrix_right->data[k * rcol + j]);
                }
            }
        }
        return matrix_product;
    }
    else
    {
        printf("Error in mulMatrix:matrix_left's column must equal to matrix_right's row!\n");
        return NULL;
    }
}

//访存优化+openMP优化
Matrix *matmul_improved2(Matrix *matrix_left, Matrix *matrix_right)
{
    if (matrix_left == NULL || matrix_right == NULL)
    {
        printf("Error in mulMatrix:one of the matrix does not exist or has been deleted!\n");
        return NULL;
    }
    else if (matrix_left->col == matrix_right->row)
    {
        Matrix *matrix_product = createMatrix_default(matrix_product, matrix_left->row, matrix_right->col);
        float r;
        size_t lrow = matrix_left->row;
        size_t lcol = matrix_left->col;
        size_t rcol = matrix_right->col;
        size_t rrow = matrix_right->row;

        // #pragma omp parallel for num_threads(32)
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < lrow; i++)
        {
            for (size_t k = 0; k < rrow; k++)
            {
                r = matrix_left->data[i * lcol + k];
                for (size_t j = 0; j < rcol; j++)
                {
                    matrix_product->data[i * matrix_product->col + j] += (r * matrix_right->data[k * rcol + j]);
                }
            }
        }
       return matrix_product;
    }
    else
    {
        printf("Error in mulMatrix:matrix_left's column must equal to matrix_right's row!\n");
        return NULL;
    }
}

// openMP+avx指令优化
Matrix *matmul_improved3(Matrix *matrix_left, Matrix *matrix_right)
{
    if (matrix_left == NULL || matrix_right == NULL)
    {
        printf("Error in mulMatrix:one of the matrix does not exist or has been deleted!\n");
        return NULL;
    }
    else if (matrix_left->col == matrix_right->row)
    {
        Matrix *matrix_product = createMatrix_default(matrix_product, matrix_left->row, matrix_right->col);
        size_t lrow = matrix_left->row;
        size_t lcol = matrix_left->col;
        size_t rcol = matrix_right->col;
        size_t rrow = matrix_right->row;
        size_t x = rcol / 8 * 8;
        if (x == rcol)
        {
            x -= 8;
        }
        __m256 value1, value2;
        __m256 beforeAdded, added, multiply;
        float cm;
//  #pragma omp parallel for num_threads(8)
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < lrow; i++)
        {
            for (size_t k = 0; k < rrow; k++)
            {
                cm = matrix_left->data[i * lcol + k];
                if (cm == 0)
                {
                    continue;
                }
                value1[0] = cm;
                value1[1] = cm;
                value1[2] = cm;
                value1[3] = cm;
                value1[4] = cm;
                value1[5] = cm;
                value1[6] = cm;
                value1[7] = cm;
                for (size_t j = 0; j + 8 < rcol; j += 8)
                {
                    value2 = _mm256_loadu_ps(&matrix_right->data[k * rcol + j]);
                    multiply = _mm256_mul_ps(value1, value2);
                    beforeAdded = _mm256_loadu_ps(matrix_product->data + i * rcol + j);
                    added = _mm256_add_ps(beforeAdded, multiply);
                    _mm256_storeu_ps(matrix_product->data + i * rcol + j, added);
                }
                for (size_t j = x; j < rcol; j++)
                {
                    matrix_product->data[i * rcol + j] += matrix_left->data[i * lcol + k] * matrix_right->data[k * rcol + j];
                }
            }
        }
        return matrix_product;
    }
    else
    {
        printf("Error in mulMatrix:matrix_left's column must equal to matrix_right's row!\n");
        return NULL;
    }
}

// 矩阵分块
Matrix *matmul_improved4(Matrix *matrix_left, Matrix *matrix_right)
{
    if (matrix_left == NULL || matrix_right == NULL)
    {
        printf("Error in mulMatrix:one of the matrix does not exist or has been deleted!\n");
        return NULL;
    }
    else if (matrix_left->col == matrix_right->row)
    {
        Matrix *matrix_product = createMatrix_default(matrix_product, matrix_left->row, matrix_right->col);
        size_t lrow = matrix_left->row;
        size_t lcol = matrix_left->col;
        size_t rcol = matrix_right->col;
        size_t rrow = matrix_right->row;
        size_t BLOCKSIZE = 0;
        float *A, *B, *C;
        size_t n = rcol;
        A = matrix_left->data;
        B = matrix_right->data;
        C = matrix_product->data;
       if(n<1000){BLOCKSIZE = n;}
    else{
        if (n%200 ==0){BLOCKSIZE=200;}
        else if(n%160 == 0){BLOCKSIZE=160;}
        else if(n%100 == 0){BLOCKSIZE=100;}
        else if (n%80 == 0){BLOCKSIZE =80;}
        else if(n%40 == 0){BLOCKSIZE = 40;}
        else if(n%32 == 0){BLOCKSIZE = 32;}
        else if(n%24 == 0){BLOCKSIZE = 24;}
        else if(n%16 == 0){BLOCKSIZE = 16;}
        else if(n%10 == 0){BLOCKSIZE = 10;}
        else if(n%8 == 0){BLOCKSIZE = 8;}
        else if (n%5 == 0){BLOCKSIZE = 5;}
        else if (n%3 == 0){BLOCKSIZE = 3;}
       else if(n%2==0){BLOCKSIZE = 2;}
       else {
           matmul_improved2(matrix_left,matrix_right);
            return;
       }
    }
        #pragma omp parallel for num_threads(8)
        for ( int sj = 0; sj < n; sj += BLOCKSIZE )
        #pragma omp parallel for num_threads(8)
        for ( int si = 0; si < n; si += BLOCKSIZE )
        #pragma omp parallel for num_threads(8)
        for ( int sk = 0; sk < n; sk += BLOCKSIZE )
        packMatrix(n, A+si*n+sk, B+sk*n+sj,  C+si*n+sj,BLOCKSIZE);
        A=NULL;
        B=NULL;
        C=NULL;
        return matrix_product;

    }
    else
    {
        printf("Error in mulMatrix:matrix_left's column must equal to matrix_right's row!\n");
        return NULL;
    }
}

void packMatrix(int n, float *A, float *B, float *C,int BLOCKSIZE) {
    for(int i = 0; i < BLOCKSIZE; i++)

    {
        for(int j = 0; j < BLOCKSIZE; j++)

        {
            float cij = C[i*n+j];
            for(int k = 0; k < BLOCKSIZE; k++ ){
                if(BLOCKSIZE - k>=8){
                    __m256 r;
                    __m256 bt;
                    __m256 c0;
                    r = _mm256_loadu_ps(&A[i*n]+k);
                    bt = _mm256_loadu_ps(&B[j]+k*n);
                    c0 += r*bt;
                    cij += (c0[1]+c0[2]+c0[0]+c0[3]+c0[4]+c0[5]+c0[6]+c0[7]);
                    k+=7;
                    continue;
                }
                cij +=A[i*n+k] * B[k*n + j];
            }
            C[i*n+j] = cij;

        }

    }
}


//打印矩阵
void printMatrix(Matrix *matrix)
{
    if (matrix == NULL)
    {
        printf("Error in printMatrix:The matrix does not exist or has been deleted!\n");
    }
    else
    {

        for (size_t i = 0; i < (matrix->col * matrix->row); i++)
        {
            printf("%lf\t", matrix->data[i]);
            if ((i + 1) % matrix->col == 0)
                printf("\n");
        }
        printf("\n");
    }
}