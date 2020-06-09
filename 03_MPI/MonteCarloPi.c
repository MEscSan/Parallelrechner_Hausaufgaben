/* See MPiMontecarlo.c in 02_Threading directory
   This programm was just extended with an MPI-Method 
*/

#include <omp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h> 

///
void scaleUpTest(int timeout, time_t start, time_t current, int timer)
{
    if(time(&current)== timer + timeout)
    { }
}
///

//Approximate Pi using Monte-Carlo Method a described in theory-block
//Boundary condition: squared of radius 1 consindered
double piMonteCarlo(int totalSamples) {

    double d = 0, x = 0, y = 0, piApprox = 0;
    int samplesInQuadrant = 0;

    //The logical approach would be to use a shared samplesInQuadrant-variable, however these apparently reduces the speed of the multi-threaded version and delivers a poor 
    //approximation of Pi. An (inefficient) alternative to the shared variable is the splitting of the for-loop into two tasks: 
    //--> calculation of the samples (parallel) and storing their distance to (0,0) in a double-array
    //--> counting of the samples in the quadrant (serial, in order to avoid the use of a shared-variable) 
    double *samples = malloc(sizeof(double) * totalSamples);

   
    for (int i = 0; i < (int)totalSamples; i++) {
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

//Multi-Threading version to approximate Pi using Monte-Carlo Method a described in theory-block
//Boundary condition: squared of radius 1 consindered
//Each Sample is generated in a parallel thread using OpenMP
double piMonteCarlo_omp(int totalSamples) {

    double x = 0, y = 0, piApprox = 0;
    
    int samplesInQuadrant=0;

    //The logical approach would be to use a shared samplesInQuadrant-variable, however these apparently reduces the speed of the multi-threaded version and delivers a poor 
    //approximation of Pi. An (inefficient) alternative to the shared variable is the splitting of the for-loop into two tasks: 
    //--> calculation of the samples (parallel) and storing their distance to (0,0) in a double-array
    //--> counting of the samples in the quadrant (serial, in order to avoid the use of a shared-variable) 
    double *samples = malloc(sizeof(double) * totalSamples);

    #pragma omp parallel for private(x,y)
    for (int i = 0; i < (int)totalSamples; i++) {
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


//Multi-Process version to approximate Pi using Monte-Carlo Method a described in theory-block
//Boundary condition: squared of radius 1 consindered
//Each process calculates only numSamples/numProcess - Samples, but it does serially 
double piMonteCarlo_mpi(int numSamples, int numProcs)
{

    double d = 0, x = 0, y = 0, piApprox = 0;
    int samplesInQuadrant = 0;
    int numSamplesInProc = (int)(numSamples / (double)numProcs);

    for (int i = 0; i < numSamplesInProc; i++)
    {
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

    //Each process calculates only one portion of piApprox
    piApprox = (double)4 * samplesInQuadrant / (double)numSamples;
    return piApprox;
}

int main(int argc, char *argv[]){
    
    //Seed random number generator
    srand(time(NULL));

    //Benchmarking variables
    clock_t start_t, startMPI_t, end_t, endMPI_t;
    double totalSerial_t, totalOMP_t, totalMPI_t, total_t;
    double speedUp_Serial2OMP, speedUp_Serial2MPI, speedUp_OMP2MPI ;
    double efficiency_Serial2MPI, efficiency_OMP2MPI;

    //Variables for Pi-Calculation
    int numSamples = 1000;
    char input[50];
    double piApprox = 0, piApproxOMP = 0, mypi= 0, piApproxMPI=0;

    //MPI-Variables
    int myID, numProcs;

    //Initialise MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myID);

    // All not-MPI Calculations are done in Process 0
    while(1){

        if(myID==0){
            //User defined number of samples:
            printf("Enter number of samples: (0 to quit) ");
            fflush(stdout);
            fgets(input, 50, stdin);
            numSamples = atoi(input);

            // Serial MonteCarlo Pi-Approximation
            start_t = clock();
            piApprox = piMonteCarlo(numSamples);
            end_t = clock();
            totalSerial_t = ((double)end_t - start_t) / CLOCKS_PER_SEC;

            // Multi Threading Pi-Approximation
            start_t = clock();
            piApproxOMP = piMonteCarlo_omp(numSamples);
            end_t = clock();
            totalOMP_t = ((double)end_t - start_t) / CLOCKS_PER_SEC;
            
            //Broadcast number of samples to all processes in order to begin MPI-Calculation
            startMPI_t = clock(); 
        }
    
        
        MPI_Bcast((void *)&numSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(numSamples==0){
                break;
        }
        mypi = piMonteCarlo_mpi(numSamples, numProcs);

        //Add the calculations of all processes
        MPI_Reduce((void*)&mypi, (void*)&piApproxMPI,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    
         //Process 0 shows the results
        if(myID == 0){

            //Time required for the MPI-Calculation
            endMPI_t = clock();
            totalMPI_t = ((double)endMPI_t - startMPI_t) / CLOCKS_PER_SEC;

            //Print the results
            printf("\n%d samples -> serial Pi-Approximation =           %f\n", numSamples, piApprox);
            printf("%d samples -> multi threaded Pi-Approximation =     %f\n", numSamples, piApproxOMP);
            printf("%d samples -> OpenMP Pi-Approximation =             %f\n", numSamples, piApproxMPI);
            
            printf("time serial =   %f\n", totalSerial_t);
            printf("time OpenMP =   %f\n", totalOMP_t);
            printf("time MPI    =   %f\n", totalMPI_t);
            
            //Speedup = time_serial/time_multiThreading
            speedUp_Serial2OMP = totalSerial_t / totalOMP_t;
            speedUp_Serial2MPI = totalSerial_t / totalMPI_t;
            speedUp_OMP2MPI = totalOMP_t / totalMPI_t;
            
            printf("speedUp Serial vs OMP = %f\n", speedUp_Serial2OMP);
            printf("speedUp Serial vs MPI = %f\n", speedUp_Serial2MPI);
            printf("speedUp    OMP vs MPI = %f\n", speedUp_OMP2MPI);

            // Parallel Speedup efficiency = speedup/number of processors
            // Only for MPI-Speedups, since OMP actually  multithreading but not multiprocessing
            efficiency_Serial2MPI = speedUp_Serial2MPI / numProcs;
            efficiency_OMP2MPI = speedUp_OMP2MPI / numProcs;

            printf("Efficiency Serial vs MPI with %d processes =    %f\n", numProcs, efficiency_Serial2MPI);
            printf("Efficiency OMP vs MPI with %d processes =       %f\n", numProcs, efficiency_OMP2MPI);
        
        
        }
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}