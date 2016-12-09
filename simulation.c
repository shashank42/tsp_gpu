#include <stdio.h>
#include <stdlib.h>

int main()
{
    printf("Hello world!\n");
    int a,b,c,d,e,f,g,h,i=1,j=1;
    //four segments condition, corresponding to swap
    while(j*i>0)
    {
        a=rand()%1000;
        b=rand()%1000;
        c=rand()%1000;
        d=rand()%1000;
        e=rand()%1000;
        f=rand()%1000;
        g=rand()%1000;
        h=rand()%1000;
        i=a*a+b*b+c*c+d*d-e*e-f*f-g*g-h*h;
        j=a+b+c+d-e-f-g-h;
        if(i*j<0)
        {
            printf("swap\n");
            printf("i %d\n",i);
            printf("j %d\n",j);
        }
    }
    //three segments condition, corresponding to insertion
    i=1;
    j=1;
    while(i*j>0)
    {
        a=rand()%1000;
        b=rand()%1000;
        c=rand()%1000;
        d=rand()%1000;
        e=rand()%1000;
        f=rand()%1000;
        i=a*a+b*b+c*c-d*d-e*e-f*f;
        j=a+b+c-d-e-f;
        if(i*j<0)
        {
            printf("insert\n");
            printf("i %d\n",i);
            printf("j %d\n",j);
        }
    }

    //two segments condition, 2-opt
    i=1;
    j=1;
    while(i*j>0)
    {
        a=rand()%1000;
        b=rand()%1000;
        c=rand()%1000;
        d=rand()%1000;
        i=a*a+b*b-c*c-d*d;
        j=a+b-c-d;
        if(i*j<0)
        {
            printf("2-opt\n");
            printf("i %d\n",i);
            printf("j %d\n",j);
        }

    }
    return 0;
}
