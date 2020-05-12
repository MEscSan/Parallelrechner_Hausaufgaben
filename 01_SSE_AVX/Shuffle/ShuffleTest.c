#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

void sse_shuffle(float* input)
{
    __m128 a;
    __m128 b;
    __m128 result;
    a = _mm_load_ps(input);
    b = _mm_load_ps(input + 4);
    result = _mm_shuffle_ps(a,b,0xE4);
    _mm_store_ps(input,result);  
}

//Small test programm for the SSE shuffle-function
//It generates two random 4-Element vectors and shuffle their elements
//with the sse shuffle-function according to the following pattern:
//Vector1:a b c d
//Vector2:e f g h
//Shuffel:a b g h 

//The shuffle-function is controlled by an 8 bit Variable to choose the vector elements
//according to the pattern in https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=Shuffle&expand=5196,5196&techs=SSE
// In order to shuffle as expected the control variable is here 0b11100100=0xE4=228
int main(int argc, char const *argv[])
{
    // Array of three 4-Elements float-Vectors
    float *vectorArray;

    //Seed the random number generator
    srand(time(NULL));

    //OS-dependent memory-aligned alloc of memory
    #if OS_WINDOWS
    vectorArray = _aligned_malloc(sizeof(float)*8, 32);
    #elif OS_UNIXOID
    vectorArray = aligned_alloc(32, sizeof(float)*8);
    #endif

    printf("Test vectors:\n");

    for(int i = 0; i<8; i++)
    {
        vectorArray[i] = rand()%100;
        printf(" %.0f",vectorArray[i]);
        if(i==3)
        {
            printf("\n");
        }
    }

    sse_shuffle(vectorArray);

    printf("\nResult:\n");

    for(int i=0; i<4; i++)
    {
        printf(" %.0f",vectorArray[i]);
    }

    printf("\n");

    //OS-dependendt memory fun
    #if OS_WINDOWS
    _aligned_free(vectorArray);
    #elif OS_UNIXOID
    free(vectorArray);
    #endif

    return EXIT_SUCCESS;
}