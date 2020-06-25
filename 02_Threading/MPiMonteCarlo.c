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
#include <math.h>
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
        d = sqrt(x * x + y * y);

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
double piMonteCarlo_omp(long totalSamples, int workers ) {

    double piApprox = 0;
    double samplesInQuadrant = 0;

    int nThreads = 0;

    //Start the parallel fork:
    #pragma omp parallel reduction(+:samplesInQuadrant) num_threads(workers)
    {
        //Seed for thread specific random number generator
        int myID = omp_get_thread_num();
        double x = 0, y = 0, d = 0;
        int i; 

    #pragma omp for 
        for (i = 0; i < totalSamples; i++) 
        {
            //Generate 2 random coordinates (x and y) in a 1x1 square surface
            x = (double)rand_r(&myID) / RAND_MAX;
            y = (double)rand_r(&myID) / RAND_MAX;

            //Calculate square of the distance of a given sample to (0,0)
            d = sqrt(x * x + y * y); 
                                    //The square of the distance of a given sample to (0,0) is smaller than 1 for points in the quadrant
            if (d <= 1) 
            {
                samplesInQuadrant++;
            }
        }
    }
    //Leave the parallel block

    //The ratio of samples in the quadrant to the number of total samples is actually pi/4, so pi = 4*ratio
    piApprox = (double)4*samplesInQuadrant/(double)totalSamples;
    return piApprox;
}

int main(int argc, char const *argv[]){
    
    //Seed random number generator
    srand(time(NULL));
    
    //Benchmarking variables
    double start_t, end_t;
    double total_t, totalOMP_t, speedUp, scaleUp;

    long totalSamples = 1000;
    long workers = 2;
    char workersInput[50];
    double piApprox = 0, mpiApprox = 0;

    if(argc > 1){
        totalSamples = atoi(argv[1]);
    }

    while(1)
    {
        printf("Number of Threads(0 quit): \n");
        fflush(stdout);
        fgets(workersInput, 50, stdin);
        workers = atoi(workersInput);

        int cycle = 0;

        if(workers==0)
        {
            break;
        }

        while(cycle < 4)
        {

            // Serial MonteCarlo Pi-Approximation
            start_t = omp_get_wtime();
            piApprox = piMonteCarlo(totalSamples);
            end_t = omp_get_wtime();
            total_t = ((double)end_t - start_t);

            // Multi Threading Pi-Approximation
            start_t = omp_get_wtime();
            mpiApprox = piMonteCarlo_omp(totalSamples, workers);
            end_t = omp_get_wtime();
            totalOMP_t = ((double)end_t - start_t);

            printf("\n%lu samples -> serial Pi-Approximation =           %f\n", totalSamples, piApprox);
            printf("%lu samples -> multi threaded Pi-Approximation =   %f\n", totalSamples, mpiApprox);

            printf("time serial =   %f\n", total_t);
            printf("time OpenMP =   %f\n", totalOMP_t);

            //Speedup = time_serial/time_multiThreading
            speedUp = total_t / totalOMP_t;   
            printf("speedUp = %f\n", speedUp);

            cycle++;
        }
    
    }


 
    return EXIT_SUCCESS;
}