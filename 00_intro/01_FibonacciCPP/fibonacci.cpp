#include<iostream>
#include<math.h>

using namespace std;

int main(int argc, char const *argv[])
{
    int n;
    long currentfib, fib_1 = 0, fib_2 = 1;
    double exponential;
    cout << "Introduce the number of Fibonnacci iterations" << endl;
    cin >> n;
    for(int i=0; i<n; i++)
    {
        exponential = exp (i);
        if(i<2)
        {
            currentfib = i;

        }
        else
        {
            currentfib = fib_2 + fib_1;
            fib_1 = fib_2;
            fib_2 = currentfib;
        }

        cout << i << " th Fibonacci number: " << currentfib << endl;
        cout << "e^"<< i << " = " << exponential << endl;
    }
    return 0;
}