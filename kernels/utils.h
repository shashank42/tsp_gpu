/*
    columbus: Software for computing approximate solutions to the traveling salesman's problem on GPUs
    Copyright (C) 2016 Steve Bronder and Haoyan Min

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _UTILS_H_
#define _UTILS_H_
 

 
#define LINE_BUF_LEN 100

/********************************************************
* Function to check for cuda error
*  Run this on the host after performing a cuda operation.
*****************************************************/
// if NDEBUG is defined before this is called, it simply returns a blank

#ifdef NDEBUG
#define cudaCheckError()
#else
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
   exit(0); \
 }                                                                 \
}
#endif
/*****************************
* Desc: structs for coordinates and tsp meta data
* 
* coordinates is the  CODE_COORD_SECTION of the tsp file
* 
* meta_data contains the information
* NAME:
* COMMENT:
* DIMENSION:
* EDGE_WEIGHT_TYPE:
******************************/
 typedef struct coordinates {
    int id;
    float x,y;
} coordinates;


typedef struct meta_data {
    char *name; char *comment;
    int dim; char *edge_type; 
} MetaTSPData;

coordinates *location;
MetaTSPData *meta = (MetaTSPData *)malloc(sizeof(MetaTSPData)); 

/*****************************
* Desc: Reads in files that are in the tsp format
*
* Input: 
* - tsp_file_name [const char()]
*  - The name of the tsp file to retrieve data from
* 
* Based on: https://github.com/bishma-stornelli/CVRP-AOC/blob/master/TSP-TEST.V0.9/instance.c
*******************************/
void read_tsp(const char *tsp_file_name){
    FILE * tsp_file;
    char buf[LINE_BUF_LEN];
    char name[LINE_BUF_LEN];
    char EDGE_WEIGHT_TYPE[LINE_BUF_LEN];
    char comment[LINE_BUF_LEN];
    char  dim[LINE_BUF_LEN];
          	  
    
    
     
    
    tsp_file = fopen(tsp_file_name, "r");
    if (tsp_file == NULL){
        printf("Failure to find file");
        exit(EXIT_FAILURE);
    }
    if (fscanf(tsp_file,"%s", buf) <= 0) printf("Reading file failed at first line\n");
    while ( strcmp("NODE_COORD_SECTION", buf) != 0 ) {
    //printf("%s \n",buf);
	if ( strcmp("NAME", buf) == 0 ) {
	    if (fscanf(tsp_file,"%s", buf) <= 0) printf("Reading file failed at Name\n");
	    // TRACE ( printf("%s ", buf); )
	    if (fscanf(tsp_file,"%s", buf) <= 0) printf("Reading file failed at Name\n");
	    strcpy(name, buf);
	    meta -> name = strdup(name);
	    //printf("%s \n",meta->name);
	    buf[0]=0;
	}
	else if ( strcmp("NAME:", buf) == 0 ) {
	    if (fscanf(tsp_file,"%s", buf) <= 0) printf("Reading file failed at Name:\n");
	    strcpy(name, buf);
	    meta -> name = strdup(name);
	    //printf("%s \n",meta->name);
	    buf[0]=0;
	}
	else if ( strcmp("NAME :", buf) == 0 ) {
	    if (fscanf(tsp_file,"%s", buf) <= 0) printf("Reading file failed at Name:\n");
	    strcpy(name, buf);
	    meta -> name = strdup(name);
	    //printf("%s \n",meta->name);
	    buf[0]=0;
	}
	else if ( strcmp("COMMENT", buf) == 0 ){
	    if(!fgets(buf, LINE_BUF_LEN, tsp_file)) printf("Reading failed at Comment\n");
	    strcpy(comment, buf);
	    meta -> comment = strdup(comment);
	    buf[0]=0;
	}
	else if ( strcmp("COMMENT:", buf) == 0 ){
	    if(!fgets(buf, LINE_BUF_LEN, tsp_file)) printf("Reading failed at Comment:\n");
        strcpy(comment, buf);
	    meta -> comment = strdup(comment);
	    buf[0]=0;
	}
	else if ( strcmp("COMMENT :", buf) == 0 ){
	    if(!fgets(buf, LINE_BUF_LEN, tsp_file)) printf("Reading failed at Comment:\n");
        strcpy(comment, buf);
	    meta -> comment = strdup(comment);
	    buf[0]=0;
	}
	else if ( strcmp("TYPE", buf) == 0 ) {
	    if (fscanf(tsp_file,"%s", buf) <= 0) printf("Reading file failed at Type\n");
	    // TRACE ( printf("%s ", buf); )
	    if (fscanf(tsp_file,"%s", buf) <= 0) printf("Reading file failed at Type\n");
	    // TRACE ( printf("%s\n", buf); )
	    if( strcmp("TSP", buf) != 0 ) {
		fprintf(stderr,"\n Not a TSP instance in TSPLIB format !!\n");
		exit(1);
	    }
	    buf[0]=0;
	}
	else if ( strcmp("TYPE:", buf) == 0 ) {
	    if (fscanf(tsp_file,"%s", buf) <= 0) printf("Reading file failed at Name:\n");
	    // TRACE ( printf("%s\n", buf); )
	    if( strcmp("TSP", buf) != 0 ) {
		fprintf(stderr,"\n Not a TSP instance in TSPLIB format !!\n");
		exit(1);
	    }
	    buf[0]=0;
	}
	else if ( strcmp("TYPE :", buf) == 0 ) {
	    if (fscanf(tsp_file,"%s", buf) <= 0) printf("Reading file failed at Name:\n");
	    // TRACE ( printf("%s\n", buf); )
	    if( strcmp("TSP", buf) != 0 ) {
		fprintf(stderr,"\n Not a TSP instance in TSPLIB format !!\n");
		exit(1);
	    }
	    buf[0]=0;
	}
	else if( strcmp("DIMENSION", buf) == 0 ){
	    if (fscanf(tsp_file,"%s", buf) <= 0) printf("Reading file failed at Dim\n");
	    strcpy(dim, buf);
	    meta -> dim = atoi(strdup(dim));
	    buf[0]=0;
	}
	else if ( strcmp("DIMENSION:", buf) == 0 ) {
	    if (fscanf(tsp_file,"%s", buf) <= 0) printf("Reading file failed at Dim:\n");
	    strcpy(dim, buf);
	    meta -> dim = atoi(strdup(dim));
	    buf[0]=0;
	}
	else if ( strcmp("DIMENSION :", buf) == 0 ) {
	    if (fscanf(tsp_file,"%s", buf) <= 0) printf("Reading file failed at Dim:\n");
	    strcpy(dim, buf);
	    meta -> dim = atoi(strdup(dim));
	    buf[0]=0;
	}
	else if( strcmp("DISPLAY_DATA_TYPE", buf) == 0 ){
	    if(!fgets(buf, LINE_BUF_LEN, tsp_file)) printf("Reading failed at Display Data Type\n");
	    // TRACE ( printf("%s", buf); );
	    buf[0]=0;
	}
	else if ( strcmp("DISPLAY_DATA_TYPE:", buf) == 0 ) {
	    if(!fgets(buf, LINE_BUF_LEN, tsp_file)) printf("Reading failed at Display Data Type\n");
	    // TRACE ( printf("%s", buf); );
	    buf[0]=0;
	}
		else if ( strcmp("DISPLAY_DATA_TYPE :", buf) == 0 ) {
	    if(!fgets(buf, LINE_BUF_LEN, tsp_file)) printf("Reading failed at Display Data Type\n");
	    // TRACE ( printf("%s", buf); );
	    buf[0]=0;
	}
	else if( strcmp("EDGE_WEIGHT_TYPE:", buf) == 0 ){
	    if(!fgets(buf, LINE_BUF_LEN, tsp_file)) printf("Reading failed at Edge Weight Type\n");
	    strcpy(EDGE_WEIGHT_TYPE, buf);
	    meta -> edge_type = strdup(EDGE_WEIGHT_TYPE);
	    buf[0]=0;
	}
	else if( strcmp("EDGE_WEIGHT_TYPE : ", buf) == 0 ){
	    if(!fgets(buf, LINE_BUF_LEN, tsp_file)) printf("Reading failed at Edge Weight Type\n");
	    strcpy(EDGE_WEIGHT_TYPE, buf);
	    meta -> edge_type = strdup(EDGE_WEIGHT_TYPE);
	    buf[0]=0;
	}
	buf[0]=0;
	if (fscanf(tsp_file,"%s", buf) <= 0) printf("Reading file failed during read\n");
    }
    printf("edge_type: %s ", meta->edge_type);
    printf("Dimension: %d \n",meta-> dim);

    //Set the location structs size.
    location = (coordinates *)malloc(meta->dim * sizeof(coordinates));
    for (int i = 0 ; i < meta->dim ; i++ ) {
	    if(fscanf(tsp_file,"%d %f %f", &location[i].id, &location[i].x, &location[i].y) <0)
	      printf("Reading failed while scanning coordinates\n");
    }
    fclose(tsp_file);
}
// END READ TSP    


/*****************************
* Desc: Reads in csv files of trips for a starting point
*
* Input: 
* - trip_file_name [const char()]
*  - The name of the tsp file to retrieve data from
* 
* Based on: https://github.com/bishma-stornelli/CVRP-AOC/blob/master/TSP-TEST.V0.9/instance.c
*******************************/
const char* getfield(char* line, int num )
{
    const char* tok;
    for (tok = strtok(line, ",");tok && *tok;tok = strtok(NULL, ",\n"))
    {
        if (!num--)
        return tok;
    }
    return NULL;
}

void read_trip(const char *trip_file_name, unsigned int *salesman_route){
    FILE* stream = fopen(trip_file_name, "r");
    if (stream == NULL){
        printf("Failure to find stream");
        exit(EXIT_FAILURE);
    }
    char line[1024];
    if (fscanf(stream,"%s", line) <= 0) printf("Reading file failed at first line\n");
    if (fgets(line, 1024, stream) == NULL) printf("Reading file failed at first line\n");
    int i = 0;
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        salesman_route[i] = atoi(getfield(tmp, 0));
        // NOTE strtok clobbers tmp
        free(tmp);
        i++;
    }
    fclose(stream);
}


/*
void read_trip(const char *trip_file_name, unsigned int *salesman_route, int N){
    FILE * trip_file;
    char buf[LINE_BUF_LEN];
    trip_file = fopen(trip_file_name, "r");
    if (trip_file == NULL)
        exit(EXIT_FAILURE);
    if (fscanf(trip_file,"%s", buf) <= 0) printf("Reading file failed at first line\n");
    for (int i = 0; i < N + 1; i++)
        if(fscanf(trip_file,"%d %f %f", &location[i].id, &location[i].x, &location[i].y) <=0)
	      printf("Reading failed while scanning coordinates\n");
    fclose(trip_file);
}
*/

 /* Function to generate random numbers in interval
 
 input:
- min [unsigned integer(1)]
  - The minimum number to sample
- max [unsigned integer(1)]
  - The maximum number to sample
  
  Output: [unsigned integer(1)]
    - A randomly generated number between the range of min and max
    
  Desc:
  Taken from
  - http://stackoverflow.com/questions/2509679/how-to-generate-a-random-number-from-within-a-range
  
  
 */
 unsigned int rand_interval(unsigned int min, unsigned int max)
{
    int r;
    const unsigned int range = 1 + max - min;
    const unsigned int buckets = RAND_MAX / range;
    const unsigned int limit = buckets * range;

    /* Create equal size buckets all in a row, then fire randomly towards
     * the buckets until you land in one of them. All buckets are equally
     * likely. If you land off the end of the line of buckets, try again. */
    do
    {
        r = rand();
    } while (r >= limit);

    return min + (r / buckets);
}

/* COMBINE TWO STRING */

char* concat(const char *s1, const char *s2)
{
    const size_t len1 = strlen(s1);
    const size_t len2 = strlen(s2);
    char *result = (char *)malloc(len1+len2+1);//+1 for the zero-terminator
    //in real code you would check for errors in malloc here
    memcpy(result, s1, len1);
    memcpy(result+len1, s2, len2+1);//+1 to copy the null-terminator
    return result;
}



#endif //_UTILS_H_
