#include <stdio.h>
#include <stdlib.h>


int main()
{

    int i, j, k, l, N=100;  //ijkl classic counters, N the number of the cities.

    struct location
    {
        int x, y;   //city's x y coordinate, respectively
    }lct[N];

    int odr[N];   //the order the salesman travels

    for(i=0; i<N; i++)
    {
        lct[i].x=rand()%1000;
        lct[i].y=rand()%1000;  //initialize the location
        odr[i]=i;     //initialize the sequence
    }

    float dist[N][N];  //distance matrix
    for(i=0; i<N; i++)
    {
        for(j=0; j<N; j++)
        {
                dist[i][j]=(lct[i].x-lct[j].x)*(lct[i].x-lct[j].x)+(lct[i].y-lct[j].y)*(lct[i].y-lct[j].y);  //calculate distance from location
                //because there'll be error when sqrt(0), so here I just calculate the square form and it also works.
          //      printf("%d %d %f\n",i,j,dist[i][j]);
        }
    }

    float sum=0;
    for(i=0;i<N;i++)
    {
        sum+=dist[odr[i]][odr[(i+1)%N]];
    }
    printf("before optimization %f\n", sum);

    float T=1000, beta=0.9999, f, a=1, b=1, delta, p;  //beta is Temp decay rate

    while(T>1)
    {
        i=rand()%N;     //city A to swap
        f=exp(-a/T);     //f is the bound parameter of swap, a is a parameter
        j=1+rand()%(int)floor(1+N*f);
        k=(i+j)%N;      //city B to swap
        delta=dist[odr[(i-1+N)%N]][odr[k]]+dist[odr[k]][odr[(i+1)%N]]+dist[odr[(k-1+N)%N]][odr[i]]+dist[odr[i]][odr[(k+1)%N]]-dist[odr[(i-1+N)%N]][odr[i]]-dist[odr[(i)]][odr[(i+1)%N]]-dist[odr[(k-1+N)%N]][odr[k]]-dist[odr[k]][odr[(k+1)%N]];
        //delta stands for change of loss function, every order should mod N in case of overflow
        p=exp(-delta*b/T);  //acceptance probability, if energy lowers, probability > 1, absolutely accept, so no extra judge needed. and b is a parameter to adjust acceptance rate
        if(p>rand()/2147483647.0)  //it's because the range of rand is 214748367, we normalize it so the result it a uniform random number between 0,1
        {
            l=odr[i];
            odr[i]=odr[k];
            odr[k]=l;
            T=T*beta;
       //     printf("%f\n",delta);
        }
    }

    sum=0;
    for(i=0;i<N;i++)
    {
        sum+=dist[odr[i]][odr[(i+1)%N]];
    }
    printf("after optimization %f\n", sum);

    return 0;
}
