#include<stdio.h>
#include<pthread.h>
#include<unistd.h>
#include<stdlib.h>
#include<time.h>

#define WAIT_TIME 1000

pthread_mutex_t accountLock, calenderLock;
int year_global = 1, month_global= 1, day_global = 0, account = 0;

void* GrannyFunction(void* data){
	
	while(1){

		pthread_mutex_lock(&calenderLock);

		int year = year_global;
		int month = month_global;
		int day = day_global;
		
		pthread_mutex_unlock(&calenderLock);

		pthread_mutex_lock(&accountLock);	
		
		//Granny looks in calender if we have the first day in the month
		if(day==1){
			//Granny gives Timmy 100 EUR
			account+=100;
			printf("%d. %d. %d -> Granny gives Timmy 100 Euro => Currently %d EUR in Timmys account\n",day,month,year, account);
		}

		pthread_mutex_unlock(&accountLock);

		usleep(WAIT_TIME);

	}
}

void* TimmyFunction(void* data){
	while(1){
		pthread_mutex_lock(&calenderLock);

		int year = year_global;
		int month = month_global;
		int day = day_global;

		pthread_mutex_unlock(&calenderLock);

		pthread_mutex_lock(&accountLock);
		
		// Timmy's age is calculated using 30-years-cycles beginning from 2002
		// One year is assumed to have 360 days
		int totalNumberOfDays = day + (month-1)*30 + (year-1)*360;
		int expenses = 30;	

		
		if(totalNumberOfDays%7==0){

			if(year > 17){
			// Now that Timmy is (at least legally) a grownup, he needs 45 EUR a week
				expenses = 45;
			}

			if(account>expenses){
				account-= expenses;
				printf("%d. %d. %d -> Timmy takes %d Euro => Currently %d EUR in Timmys account\n", day, month, year, expenses, account);
			}
			else{
				printf("%d. %d. %d -> Timmy cannot take any money => Currently %d EUR in Timmys account\n", day, month, year,account);
			}
		
		}
		pthread_mutex_unlock(&accountLock);
		usleep(WAIT_TIME);
	}
}

void* TimmysRiderFunction(void* data){
	while(1){
		pthread_mutex_lock(&calenderLock);

		int year = year_global;
		int month = month_global;
		int day = day_global;

		pthread_mutex_unlock(&calenderLock);

		pthread_mutex_lock(&accountLock);

		// Timmy is only allowed to work once he is over 18
		int salary = 120;

		if( year > 17 && day == 29){	
			
			//Timmy earns between 50 and 120 EUR a month with a Rider-Minijob
			//The the payday is 29th day in the month
			salary -= rand()%70;
			account += salary;
		
			printf("%d. %d. %d -> Timmy earned %d EUR => Currently %d EUR in Timmys account\n", day, month, year,salary, account);
			
		}
		pthread_mutex_unlock(&accountLock);
		usleep(WAIT_TIME);
	}
}

//Counts days and months
void* Calender(void* data){
	int year = 0;
	while (year < 30)
	{
		pthread_mutex_lock(&calenderLock);	

		day_global++;
			
		//Months are assumed to have 30 days, every 30 days the day-counter resets	
		if(day_global>30){
			day_global = 1;
			month_global++;	
		}
		
		//Every 12 months the month-counter resets			
		if(month_global>12){
			month_global = 1;
			year_global++;	
		}

		year = year_global;
		pthread_mutex_unlock(&calenderLock);
		usleep(WAIT_TIME);
	}
}

int main(int argc, char const *argv[]){

	pthread_t calender, granny, timmy, timmysRiderJob;	
	
	//Seed for the random number generator
	srand(time(NULL));
	
	pthread_mutex_init(&calenderLock, NULL);
	pthread_mutex_init(&accountLock, NULL);
	//Create calender, granny- timmy and timmysRiderJob-threads, no input parameters needed
	pthread_create(&calender, NULL, Calender, NULL);
	pthread_create(&granny, NULL, GrannyFunction, NULL);
	pthread_create(&timmy, NULL, TimmyFunction, NULL);
	pthread_create(&timmysRiderJob, NULL, TimmysRiderFunction, NULL);
	pthread_join(calender,NULL);
	return EXIT_SUCCESS;
}
