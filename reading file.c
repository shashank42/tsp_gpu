#include <stdio.h>
#include <stdlib.h>


int main()
{
    FILE *fp;
        fp=fopen("ch130.tsp","r");   //open file,read only
    int i;
    struct {
        float x;
        float y;
    }c[130];       //130 cities
    fscanf(fp,"%d",&i);
    while(i!=1)           //keep moving the pointer fp until it points to index 1
    {
        fseek(fp,1,SEEK_CUR);
        fscanf(fp,"%d",&i);
    }
    fscanf(fp,"%f",&c[0].x);
    printf("%f\n",c[0].x);
    fscanf(fp,"%f",&c[0].y);
    printf("%f\n",c[0].y);
    while(i!=130)
    {
        fscanf(fp,"%d",&i);   //index
        printf("%d\n",i);
        fscanf(fp,"%f",&c[i-1].x);  //x
        printf("%f ",c[i-1].x);
        fscanf(fp,"%f",&c[i-1].y);  //y
        printf("%f\n",c[i-1].y);
    }

    fclose(fp);

    return(0);
}
