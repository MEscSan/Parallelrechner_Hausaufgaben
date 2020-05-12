#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//intrinsics
#include <immintrin.h>

// Comparison of matrix product with a naive approach and using intrinsics

// Typical matrix multipliclation (naive-Algorithm from Wikipedia:https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm)
// For-loops slightly adapted for vectorwise matrix storage
// Matrix A(see algorithm) stored in row-major-order
// Matrix B(see algorithm) stored in col-major-order
void matMult(float* matA, float* matB, float* matC, int rowsA, int colsA, int rowsB, int colsB)
{
    //check if matrix multiplication possible:
    if(colsA != rowsB)
    {
        printf("Wrong matrix dimensions!\n");
    }
    else
    {
        int numberOfAElements = rowsA*colsA;
        int numberOfBElements = rowsB*colsB;
        int indexC = 0;
        int sum = 0;

        for( int i=0; i < numberOfAElements; i += colsA )
        {
            for( int j=0; j < numberOfBElements; j += rowsB)
            { 
                sum = 0;

                for(int k=0; k < colsA; k++)
                {
                    sum += matA[i+k] * matB[j+k];  
                }

                matC[indexC] = sum;
                indexC++;
            }
        }
    }
}

// SSE matrix multipliclation: naive-algorithm with SIMD element-calculation
// No parallelization of the matrix-product as such; instead partial 
// parallelization of the dot-product used for element calculation
// Matrix A(see algorithm) stored in row-major-order
// Matrix B(see algorithm) stored in col-major-order
void matMultSSE(float* matA, float* matB, float* matC, int rowsA, int colsA, int rowsB, int colsB)
{
    //check if matrix multiplication possible:
    if (colsA != rowsB)
    {
        printf("Wrong matrix dimensions!\n");
    }
    else
    {
        int numberOfAElements = rowsA * colsA;
        int numberOfBElements = rowsB * colsB;
        int indexC = 0;
        float tmp[4];
        __m128 sum;
        __m128 vectorA;
        __m128 vectorB;
        
        for (int i = 0; i < numberOfAElements; i += colsA)
        {
            for (int j = 0; j < numberOfBElements; j += rowsB)
            {
                sum = _mm_setzero_ps();

                // SIMD element calculation using dot product 
                // (SIMD-Dot Product implementation, see:https://lemire.me/blog/2018/07/05/how-quickly-can-you-compute-the-dot-product-between-two-large-vectors/ )
                for (int k = 0; k < colsA; k+=4)
                {
                    //Load first 4 elements of the row(A)/column(B) in an SSE-Register
                    vectorA = _mm_load_ps(matA + i + k);
                    vectorB = _mm_load_ps(matB + j + k);

                    //Dot product using SSE-Registers
                    sum = _mm_add_ps(sum, _mm_mul_ps(vectorA,vectorB));

                }                   
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);

                //Store element in the C-Matrix
                _mm_store_ps(tmp, sum);
                matC[indexC] = tmp[0];
                indexC++;
            }
        }
    }
}

// AVX matrix multipliclation: naive-algorithm with SIMD element-calculation
// No parallelization of the matrix-product as such; instead partial 
// parallelization of the dot-product used for element calculation
// Matrix A(see algorithm) stored in row-major-order
// Matrix B(see algorithm) stored in col-major-order
void matMultAVX(float* matA, float* matB, float* matC, int rowsA, int colsA, int rowsB, int colsB)
{
    //check if matrix multiplication possible:
    if (colsA != rowsB)
    {
        printf("Wrong matrix dimensions!\n");
    }
    else
    {
        int numberOfAElements = rowsA * colsA;
        int numberOfBElements = rowsB * colsB;
        int indexC = 0;
        float tmp[8];
        __m256 sum;
        __m256 vectorA;
        __m256 vectorB;

        for (int i = 0; i < numberOfAElements; i += colsA)
        {
            for (int j = 0; j < numberOfBElements; j += rowsB)
            {
                sum = _mm256_setzero_ps();

                // SIMD element calculation using dot product 
                //(SIMD-Dot Product implementation, see:https://lemire.me/blog/2018/07/05/how-quickly-can-you-compute-the-dot-product-between-two-large-vectors/ )
                for (int k = 0; k < colsA; k += 8)
                {
                    //Load first 8 elements of the row(A)/column(B) in an AVX-Register
                    vectorA = _mm256_load_ps(matA + i + k);
                    vectorB = _mm256_load_ps(matB + j + k);

                    //Dot product using SSE-Registers
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(vectorA, vectorB));
                }

                sum = _mm256_hadd_ps(sum, sum);
                sum = _mm256_hadd_ps(sum, sum);
                
                //Store element in the C-Matrix
                _mm256_store_ps(tmp, sum);
                matC[indexC] = tmp[0]+tmp[4];
                indexC++;
            }
        }
    }
}


//Compares elementwise two matrices of the same dimensions
//Output:1 if all elements are equal, 0 else
int equalMat(float* matA, float* matB, int rows, int cols) 
{
    int eqMat = 1;
    int numberOfElements = rows * cols;
    for (int i = 0; i < numberOfElements; i++)
    {
        if (matA[i] != matB[i]) 
        {
            eqMat = 0;
        }
    }

    return eqMat;
}

// Display a matrix in Col-Major order
void printMatColMajor(float* mat, int rows, int cols)
{
    int numberOfElements = rows*cols;
    for(int i=0; i < rows; i++)
    {
        printf("[");
        for(int j=0; j < cols; j++)
        {
            printf(" %.0f", mat[i+j*rows]);
        }
        printf(" ]\n");
    }
}

// Display a matrix in Row-Major order
void printMatRowMajor(float* mat, int rows, int cols)
{
    int numberOfElements = rows*cols;
    for(int i=0; i < numberOfElements; i += cols)
    {
        printf("[");
        for(int j=0; j < cols; j++)
        {
            printf(" %.0f", mat[i+j]);
        }
        printf(" ]\n");
    }
}

// Generate a random matrix given its dimensions
void randMat(float* mat, int rows, int cols)
{
    int numberOfElements = rows * cols;
    for (int i = 0; i < numberOfElements; i++)
    {
        mat[i] = rand() % 10;
    }
}


int main(int argc, char const *argv[]) 
{
    clock_t start_t, end_t;
    double total_t;
    float *matA, *matB, *matC, *matC_SSE, *matC_AVX;
    int rowsA = 0, colsA = 0, rowsB = 0, colsB = 0, rowsC = 0, colsC = 0;
    int right_SSE = 0, right_AVX = 0;

    //Seed random number generator
    srand(time(NULL));

    //allow matrix-dimensions input in command-line
    if(argc < 2)
    {
        rowsA = 1024;
        colsA = 1024;
        rowsB = 1024;
        colsB = 1024;
    }
    else
    {
        rowsA = atol(argv[1]);
        colsA = atol(argv[2]);
        rowsB = atol(argv[3]);
        colsB = atol(argv[4]);

        //For simplicity: Matrix dimensions have to be mutiple of 8 (to fit in both SSE and AVX-Registers)
        //Round user-input matrix-dimensions to the next bigger multiple of 8
        if (rowsA % 8 != 0) 
        {
            rowsA = (1 + (rowsA/8))*rowsA;
        }
        if (colsA % 8 != 0)
        {
            colsA = (1 + (colsA / 8)) * colsA;
        }
        if (rowsB % 8 != 0)
        {
            rowsB = (1 + (rowsB / 8)) * rowsB;
        }
        if (colsB % 8 != 0)
        {
            colsB = (1 + (rowsA / 8)) * colsB;
        }

    }

    rowsC = rowsA;
    colsC = colsB;

    //OS-dependent memory-alligned alloc of memory
    #if OS_WINDOWS
    matA = _aligned_malloc(sizeof(int) * rowsA * colsA, 32);
    matB = _aligned_malloc(sizeof(int) * rowsB * colsB, 32);
    matC = _aligned_malloc(sizeof(int) * rowsC * colsC, 32);
    matC_SSE = _aligned_malloc(sizeof(int) * rowsC * colsC, 32);
    matC_AVX = _aligned_malloc(sizeof(int) * rowsC * colsC, 32);
    #elif OS_UNIXOID
    matA = aligned_alloc(32, sizeof(int) * rowsA * colsA);
    matB = aligned_alloc(32, sizeof(int) * rowsB * colsB);
    matC = aligned_alloc(32, sizeof(int) * rowsC * colsC);
    matC_SSE = aligned_alloc(32, sizeof(int) * rowsC * colsC);
    matC_AVX = aligned_alloc(32, sizeof(int) * rowsC * colsC);
    #endif

    //Generate random Matrices:
    randMat(matA, rowsA, colsA);
    randMat(matB, rowsB, colsB);
    // Show matrices_
    printf("Matrix A (Dim.: %d x %d)\n", rowsA, colsA);
    //printMatRowMajor(matA, rowsA, colsA);
    printf("Matrix B (Dim.: %d x %d)\n", rowsB, colsB);
    //printMatColMajor(matB, rowsB, colsB);

    //Benchmarking

    //No SIMD
    start_t = clock();
    matMult(matA, matB, matC, rowsA, colsA, rowsB, colsB);
    end_t = clock();
    total_t = ((double)end_t - (double)start_t) / CLOCKS_PER_SEC;
    printf("\nNormal: %f\n", total_t);
    //printf("\nNormal matrix product: \n");
    //printMatRowMajor(matC, rowsC, colsC);
    
    //SSE
    start_t = clock();
    matMultSSE(matA, matB, matC_SSE, rowsA, colsA, rowsB, colsB);
    end_t = clock();
    total_t = ((double)end_t - (double)start_t) / CLOCKS_PER_SEC;
    printf("\nSSE: %f\n", total_t);
    right_SSE = equalMat(matC, matC_SSE, rowsC, colsC);
    if (right_SSE) 
    {
        printf("SSE Matrix product was right\n");
    }
    
    //AVX
    start_t = clock();
    matMultAVX(matA, matB, matC_AVX, rowsA, colsA, rowsB, colsB);
    end_t = clock();
    total_t = ((double)end_t - (double)start_t) / CLOCKS_PER_SEC;
    printf("\nAVX: %f\n", total_t);
    right_AVX = equalMat(matC, matC_AVX, rowsC, colsC);
    if (right_AVX)
    {
        printf("AVX Matrix product was right\n");
    }
    

    //OS-dependendt memory fun
    #if OS_WINDOWS
    _aligned_free(matA);
    _aligned_free(matB);
    _aligned_free(matC);
    _aligned_free(matC_SSE);
    _aligned_free(matC_AVX);
    #elif OS_UNIXOID
    free(matA);
    free(matB);
    free(matC);
    free(matC_SSE);
    free(matC_AVX);
    #endif

    // exit
    return EXIT_SUCCESS;
}