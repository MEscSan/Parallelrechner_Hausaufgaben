#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include<time.h>
#include<sys/time.h>
#include<math.h>
#include<omp.h>
#include<mpi.h>

//Source Code originally written for:
//          OS: Ubuntu 18.04.1
//          CPU: Intel Core i7-5500U 
#define OS "Ubuntu 18.04.1"
#define CPU "Inte Core i7-5500U"


//Small Integral and Power-Series calculations:
// -> One Process to approximate the solution, the rest to approximate the integral
// -> Natural Logarithm approximated using Power series
// -> Pi approximated using Monte-Carlo Algorithm from previous lessons
// -> Parameters: number of Threads, number of iterations (for both Monte Carlo Approximation and power series as well as Integral intervals)

//Parallel - Power series for ln(x)
double ln_powerSeries_parallel(double x, long long iterations, int workers)
{
    double ln_x = 0;

    // Start the fork of threads
    #pragma omp parallel reduction(+: ln_x) num_threads(workers)
    {
        long long n;
        
        #pragma omp for 
        for(n = 1; n < iterations; ++n)
        {
            ln_x += pow(-1, n + 1) * pow(x - 1, n) / (double)n;
        }
    }

    return ln_x;
}

double ln_powerSeries_sequential(double x, long long iterations)
{
    double ln_x = 0;
    long long n;
        
    for(n = 1; n < iterations; n++)
    {
        ln_x += pow(-1, n + 1) * pow(x - 1, n) / (double)n;
    }
    
    return ln_x;
}

//Parallel- Monte Carlo - Pi
double pi_parallel(long long iterations, int workers, int numProcs)
{
    double pi = 0;
    long long samplesInQuadrant = 0;

    // Start the fork of threads
    #pragma omp parallel reduction(+: samplesInQuadrant) num_threads(workers)
    {
        //Seed for random number generator for each thread
        int mySeed = omp_get_thread_num();

        double x = 0;
        double y = 0;
        double d = 0;
        long long i;

        #pragma omp for 
        for(i = 0; i < iterations; i+=numProcs)
        {
            x = (double)rand_r(&mySeed) / RAND_MAX;
            y = (double)rand_r(&mySeed) / RAND_MAX;

            d = x * x + y * y;

            if(d <= 1)
            {
                ++samplesInQuadrant;
            }
        }        
    }

    pi = (double)4*samplesInQuadrant/(double)iterations;
    return pi;
}

double pi_sequential(long long iterations)
{
    double pi = 0;
    long long samplesInQuadrant = 0;

    srand((int)time(NULL));

    double x = 0;
    double y = 0;
    double d = 0;
    long long i;

        
    for(i = 0; i < iterations; i++)
    {
        x = ((double)rand()) / RAND_MAX;
        y = ((double)rand()) / RAND_MAX;
        d = x * x + y * y;

        if(d <= 1)
        {
            samplesInQuadrant++;
        }        
    }

    pi = (double)4*samplesInQuadrant/(double)iterations;
    return pi;
}

// MPI - $\int_{0}^{1}\frac{ln(1+x)}{x²+1}dx$
double integral_parallel(long long intervals, long long iterations, int numProcs, int workers, int myId)
{
    //Integral approximation with a Riemann-Sum (asuming equal intervals)
    double h = 1.0 / intervals;
    long long n;
    double x = 0;
    double ln = 0;
    double value = 0;

    //Riemann-Summ
    #pragma omp parallel reduction(+: value) num_threads(workers)
    {
        #pragma omp for  
        for (n = myId; n < intervals; n += numProcs)
        {
            x = h*n;
            ln = ln_powerSeries_parallel(1 + x, iterations, workers);
            value += ln / (x * x + 1);
        }
    }
    
    value = h * value;

    return value;

}

double integral_sequential(long long intervals, long long iterations)
{

    //Integral approximation with a Riemann-Sum (asuming equal intervals)
    double h = 1.0 / intervals;
    long long n = 1;
    double x = 0;
    double ln = 0;
    double value = 0;

    for (n; n < intervals; n++)
    {
        x += h;
        ln = ln_powerSeries_sequential(1 + x, iterations);
        value += ln / (x * x + 1);
    }

    value = h * value;

    return value;
}

double integral_c(long long intervals)
{
    double h = 1.0 / intervals;
    long long n = 1;
    double x = 0;
    double ln = 0;
    double value = 0;

    for (n; n < intervals; n++)
    {
        x += h;
        ln = log(1 + x);
        value += ln / (x * x + 1);
    }

    value = h * value;

    return value;
}

// pi*ln(2)/8
double value_parallel(long long iterations, int workers, int numProcs)
{
    double pi = pi_parallel(iterations, workers, numProcs);
    double ln2 = ln_powerSeries_parallel(2, iterations, workers);
    double value = pi * ln2 / (double)8;

    return value;
}

double value_sequential(long long iterations)
{
    double pi = pi_sequential(iterations);
    double ln2 = ln_powerSeries_sequential(2, iterations);
    double value = pi * ln2 / (double)8;
    return value;
}

int main(int argc, char* argv[])
{
    
    int myId, value, numProcs;
    int done = 0;
    int cycle = 0;

    MPI_Init(&argc, &argv);
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    long long iterations = 1000;
    char iterationsInput[50];
    long long intervals = 1000;
    char intervalsInput[50];
    int workers = 4;
    char workersInput[10];

    double value_par, value_proc, value_seq, value_c;
    double integral_proc, integral_par, integral_seq, integral_cMath;
    double start_parallel, start_sequential, start_c;
    double end_parallel, end_sequential, end_c;
    double time_parallel, time_sequential, time_c;

    if(myId == 0)
    {
        printf("Warning: source code originally written for\n\t\tOS: %s\n\t\tCPU: %s\n", OS, CPU);
    }

    //In order to get stable results calculations are done four times
    //  ->  1 Warm up
    //  ->  3 Benchmarking  
    while(!done)
    {
        if(myId == 0)
        {
            if(cycle == 0)
            {

                printf("Number of series iterations:(0 to quit) \n");
                fflush(stdout);
                fgets(iterationsInput, 50, stdin);
                iterations = atoll(iterationsInput);

                printf("Number of integration intervals: \n");
                fflush(stdout);
                fgets(intervalsInput, 50, stdin);
                intervals = atoll(intervalsInput);

                printf("Maximal number of threads: %d\n", omp_get_max_threads()) ;

                printf("Number of Open MP threads: \n");
                fflush(stdout);
                fgets(workersInput, 10, stdin);
                workers = atoi(workersInput);

            }


            //start_sequential = clock();
            start_sequential = MPI_Wtime();

            value_seq = value_sequential(iterations);
            integral_seq = integral_sequential(intervals, iterations);

            end_sequential = MPI_Wtime();

            start_c = MPI_Wtime();

            value_c = M_PI * log(2) / (double)8;
            integral_cMath = integral_c(intervals);

            end_c = MPI_Wtime(); 
            
            start_parallel = MPI_Wtime();

        }

        MPI_Bcast((void *)&iterations, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast((void*)&intervals, 1,  MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);    
        MPI_Bcast((void*)&workers, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if(iterations == 0)
            break;

        value_proc = value_parallel(iterations, workers, numProcs);
        MPI_Reduce((void*)&value_proc, (void*) &value_par, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        integral_proc = integral_parallel(intervals, iterations, numProcs, workers, myId);

        MPI_Reduce((void *)&integral_proc, (void*) &integral_par, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if(myId == 0)
        {
          end_parallel = MPI_Wtime();

            time_c = (end_c - start_c);
            time_sequential = (end_sequential - start_sequential);
            time_parallel = end_parallel - start_parallel;

            if(cycle==0)
            {
                printf("Warm up - Round\n");
            }

            printf("C Math-Functions    :   Value = %.10f  Integral = %.10f Time = %f\n", value_c, integral_cMath, time_c);
            fflush(stdout);
            printf("Sequential          :   Value = %.10f  Integral = %.10f Time = %f\n", value_seq, integral_seq, time_sequential);
            fflush(stdout);
            printf("Parallel            :   Value = %.10f  Integral = %.10f Time = %f\n\n", value_par, integral_par, time_parallel);
            fflush(stdout);

            if(cycle < 4)
            {
                cycle++;
            }
            else
            {
                cycle = 0;
            }
        
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}