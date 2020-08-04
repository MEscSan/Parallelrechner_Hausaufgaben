#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CudaErrorHelper.h"

using namespace std;

__global__
void linearFilter(int rows, int cols, char* pixel_ptr)
{
    // TODO Linear-Filter Implementation
}

__host__
// Parse a string in a file to an integer
// till a whitespace (ASCII Code 32) found
int parseNumber(FILE * file)
{
    unsigned char current_char = fgetc(file);
    int number = 0;
    do
    {
        printf("%d\n", current_char);
        usleep(100000);
        // The first character might be a whitespace
        // and it should be ignored
        if(current_char != 32)
        {
            number *= 10;
            // "Padding" ASCII-Decimal code to figures (0-9)
            number += (current_char - 48);
        }

        current_char = fgetc(file) ;
    }
    while(current_char != 32);

    printf("%d\n", number);
    return number;
}

__host__
// Parse a string in a file to an integer
// till a whitespace (ASCII Code 32) found
int parseNumber(FILE * file, int numChars)
{
    unsigned char current_char = fgetc(file);
    int number = 0;
    do
    {
        printf("%d\n", current_char);
        usleep(100000);
        // The first character might be a whitespace
        // and it should be ignored
        if(current_char != 32)
        {
            number *= 10;
            // "Padding" ASCII-Decimal code to figures (0-9)
            number += (current_char - 48);
        }

        current_char = fgetc(file) ;
    }
    while(current_char != 32);

    printf("%d\n", number);
    return number;
}

int main(void)
{   
    FILE * img = fopen("../lena.ppm", "r");    
    //char imgHeader[15];
    fgetc(img);

    //First read the image format (the first 2 bytes)
    char format = fgetc(img);
    int rows = 0, cols=0, numColors=0;

    switch(format)
    {
        case '6': 
            cout << " Binary Portable PixMap(.ppm)\n";
            cols = parseNumber(img);
            rows = parseNumber(img);
            fscanf(img,"%d\n",&numColors);
            //numColors = parseNumber(img);
            break;
        default:   cout << "Could not identify image format\n";
            break;
    }


    printf("File Header:\n rows: %d\n cols: %d\n Number of Colors: %d\n", rows, cols, numColors);
    FILE * img_dst = fopen ("lena_ppm_pixels.txt", "w");
 
    for(int i = 0; i < rows; i++)
    {
        char row[512];
        fgets(row, cols, img);
        fwrite ((unsigned char*)row, sizeof(char), sizeof(row), img_dst);
        for(int j = 0; j < cols; j++)
        {
            
            printf("%u\t",(unsigned char)row[j]);
            
        }
        printf("\nEnd\n");

    }
    fclose (img_dst);
    return EXIT_SUCCESS;
}


