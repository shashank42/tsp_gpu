#include <stdio.h>
#include <stdlib.h>

int main()
{
    printf("Hello world!\n");
    int a,b,c,d,e,f,g,h,i=1,j=1,k=0,l=0,m=2000;
    //l counts how many wrong we made while using square
    //m is distance range
    //four segments condition, corresponding to swap
    for(m=100;m<10000;m*=2)
    {
    i=1;
    j=1;
    k=0;
    l=0;
    printf("range %d\n",m);
    while(k<1000000)
    {
        a=rand()%m;
        b=rand()%m;
        c=rand()%m;
        d=rand()%m;
        e=rand()%m;
        f=rand()%m;
        g=rand()%m;
        h=rand()%m;
        i=a*a+b*b+c*c+d*d-e*e-f*f-g*g-h*h;
        j=a+b+c+d-e-f-g-h;
        if(i*j<0)
        {
            l++;
        }
        k++;
    }
    //probability of wrong acceptance
    printf("4seg %f\n",l/1000000.0);
    //three segments condition, corresponding to insertion
    i=1;
    j=1;
    k=0;
    l=0;
    while(k<1000000)
    {
        a=rand()%m;
        b=rand()%m;
        c=rand()%m;
        d=rand()%m;
        e=rand()%m;
        f=rand()%m;
        i=a*a+b*b+c*c-d*d-e*e-f*f;
        j=a+b+c-d-e-f;
        if(i*j<0)
        {
            l++;
        }
        k++;
    }
    printf("3seg %f\n",l/1000000.0);

    //two segments condition, 2-opt
    i=1;
    j=1;
    k=0;
    l=0;
    while(k<1000000)
    {
        a=rand()%m;
        b=rand()%m;
        c=rand()%m;
        d=rand()%m;
        i=a*a+b*b-c*c-d*d;
        j=a+b-c-d;
        if(i*j<0)
        {
            l++;
        }
        k++;
    }
    printf("2seg %f\n",l/1000000.0);
    }
    return 0;
}
