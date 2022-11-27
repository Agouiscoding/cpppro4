#ifndef __SIZE_T
#define __SIZE_T
typedef unsigned long size_t;
#endif
#pragma GCC optimize(3)
#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math")
#pragma once

typedef struct
{
    size_t row;
    size_t col;
    float *data;
} Matrix;

Matrix *createMatrix_default(Matrix *matrix, size_t row, size_t col);               //声明一个m*n的矩阵并初始化为0
Matrix *createMatrix_initial(Matrix *matrix, size_t row, size_t col, float *array); //声明一个矩阵并用数组初始化
void valueMatrix_array(Matrix *matrix, float *array);                               //用数组给矩阵赋值
void valueMatrix_console(Matrix *matrix);                                           //用控制台给矩阵赋值
void deleteMatrix(Matrix **matrix);                                                 //删除一个矩阵 释放空间 指针置NULL
size_t sizeMatrix(Matrix *matrix);                                                  //返回矩阵的大小row*col
Matrix *matmul_plain(Matrix *matrix_left, Matrix *matrix_right);                    //朴素的矩阵相乘
Matrix *matmul_improved1(Matrix *matrix_left, Matrix *matrix_right);                ////访存优化
Matrix *matmul_improved2(Matrix *matrix_left, Matrix *matrix_right);                ////访存优化+openMP优化
Matrix *matmul_improved3(Matrix *matrix_left, Matrix *matrix_right);               // openMP+avx指令优化
Matrix *matmul_improved4(Matrix *matrix_left, Matrix *matrix_right);                 // 矩阵分块
void printMatrix(Matrix *matrix);                                                   //打印矩阵