/*
    OpenMP-Test project to approximate the number Pi using a Monte-Carlo Method

    Theory: 
    ->  Fork-Join Parallelism: Master-thread branches at certain points into a team of threads (fork)
        that once finished join back to the Master-Thread
        Source: https://en.wikipedia.org/wiki/Fork%E2%80%93join_model

    ->  OpenMP Version for this Code: Visual Studio 16 2019 => OpenMP 2.0 Standard

    ->  Compiler-Directives/-Clauses
            barrier:Synchronizes all threads in a team; all threads pause at the barrier, untill all 
                    threads execute the barrier

            critical:Specifies that code is only executed on one thread at a time

            nowait: Overrides the barrier implicit in another directive
            
            private:Specifies that each Thread should have its own instance of a Variable
            
            reduction:Specifies that one or more variables that are private to each thread are reduced
                    to a single one via a reduction operation at the end of the parallel region
            
            shared:Specifies that one or more variables should be shared among all threads
        
            single: Allows to specify tha a section of code should be executed on a single thread,
                    no necesarily the master thread
            
        Source: 
        https://docs.microsoft.com/en-us/cpp/parallel/openmp/reference/openmp-directives?view=vs-2019
    
    ->  Approximation of Pi using Monte-Carlo Method: 
            1.  Consider a square of side r (Asumption: s = 1) with inscribed quadrant
            2.  Generate a number n of uniformly distributed sample-points in the square
            3.  Count number of samples inside and outside the quadrant
            4.  Pi = 4*(Number of samples inside quadrant/Number of total samples)

        Source:https://en.wikipedia.org/wiki/Monte_Carlo_method
*/
#include <omp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>


//Approximate Pi using Monte-Carlo Method a described in theory-block
//Boundary condition: squared of radius 1 consindered
double piMonteCarlo(int totalSamples) {

    double d = 0, x = 0, y = 0, piApprox = 0;
    int samplesInQuadrant = 0;

    for (int i = 0; i < totalSamples; i++) {
        //Generate 2 random coordinates (x and y) in a 1x1 square surface
        x = ((double)rand()) / RAND_MAX;
        y = ((double)rand()) / RAND_MAX;

        //Calculate square of the distance of a given sample to (0,0)
        d = x * x + y * y;

        //The square of the distance of a given sample to (0,0) is smaller than 1 for points in the quadrant
        if (d <= 1) {
            samplesInQuadrant++;
        }
    }

    //The ratio of samples in the quadrant to the number of total samples is actually pi/4, so pi = 4*ratio
    piApprox = (double)4 * samplesInQuadrant / (double)totalSamples;
    return piApprox;
}

//Multi-Threading version to approximate Pi using Monte-Carlo Method a described in theory-block
//Boundary condition: squared of radius 1 consindered
//Each Sample is generated in a parallel thread using OpenMP
double piMonteCarlo_omp(int totalSamples) {

    double x = 0, y = 0, piApprox = 0;
    int i;
    int samplesInQuadrant=0;

    //The logical approach would be to use a shared samplesInQuadrant-variable, however these apparently reduces the speed of the multi-threaded version and delivers a poor 
    //approximation of Pi. An (inefficient) alternative to the shared variable is the splitting of the for-loop into two tasks: 
    //--> calculation of the samples (parallel) and storing their distance to (0,0) in a double-array
    //--> counting of the samples in the quadrant (serial, in order to avoid the use of a shared-variable) 
    double *samples = malloc(sizeof(double) * totalSamples);

    #pragma omp parallel for private(x,y)
    for (i = 0; i < (int)totalSamples; i++) {
        //Generate 2 random coordinates (x and y) in a 1x1 square surface
        x = ((double)rand()) / RAND_MAX;
        y = ((double)rand()) / RAND_MAX;

        //Calculate square of the distance of a given sample to (0,0)
        samples[i] = x * x + y * y;
     }
     
    for (int i = 0; i < totalSamples; i++)
    {
        //The square of the distance of a given sample to (0,0) is smaller than 1 for points in the quadrant
        if (samples[i] <= 1) {
            samplesInQuadrant++;
        }
    }
  
    //The ratio of samples in the quadrant to the number of total samples is actually pi/4, so pi = 4*ratio
    piApprox = (double)4*samplesInQuadrant/(double)totalSamples;
    return piApprox;
}

int main(int argc, char const *argv[]){
    
    //Seed random number generator
    srand(time(NULL));
    
    //Benchmarking variables
    clock_t start_t, end_t;
    double total_t, totalOMP_t, speedUp, scaleUp;

    int totalSamples = 1000;
    double piApprox = 0, mpiApprox = 0;

    if(argc > 1){
        totalSamples = atoi(argv[1]);
    }

    // Serial MonteCarlo Pi-Approximation
    start_t = clock();
    piApprox = piMonteCarlo(totalSamples);
    end_t = clock();
    total_t = ((double)end_t - start_t) / CLOCKS_PER_SEC;

    // Multi Threading Pi-Approximation
    start_t = clock();
    mpiApprox = piMonteCarlo_omp(totalSamples);
    end_t = clock();
    totalOMP_t = ((double)end_t - start_t) / CLOCKS_PER_SEC;

    printf("\n%d samples -> serial Pi-Approximation =           %f\n", totalSamples, piApprox);
    printf("%d samples -> multi threaded Pi-Approximation =   %f\n", totalSamples, mpiApprox);

    printf("time serial =   %f\n", total_t);
    printf("time OpenMP =   %f\n", totalOMP_t);

    //Speedup = time_serial/time_multiThreading
    speedUp = total_t / totalOMP_t;

    printf("speedUp = %f", speedUp);

    return EXIT_SUCCESS;
}