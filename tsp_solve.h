#ifndef _TSP_SOLVE_H_
#define _TSP_SOLVE_H_



/* TSP With Only Difference Calculation
Input:
- city_one: [unsigned integer(GRID_SIZE)]
  - Cities to swap for the first swap choice
- city_two: [unsigned integer(GRID_SIZE)] 
  - Cities to swap for the second swap choice
- dist: [float(N * N)]
  - The distance matrix of each city
- salesman_route: [unsigned integer(N)] 
  - The route the salesman will travel
- T: [unsigned integer(1)] 
  - The current temperature
- r: [unsigned integer(GRID_SIZE)]
  - The random number to compare against for S.A.
*/
__global__ static void tsp(unsigned int* __restrict__ city_one,
                           unsigned int* __restrict__ city_two,
                           coordinates* __restrict__ location,
                           unsigned int* __restrict__ salesman_route,
                           float* __restrict__ T,
                           float* __restrict__ r,
                           unsigned int *flag,
                           volatile unsigned int *global_flag){
                           
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float delta, p;
    unsigned int city_one_swap = city_one[tid];
    unsigned int city_two_swap = city_two[tid];
    
    unsigned int trip_city_one      = salesman_route[city_one_swap];
    unsigned int trip_city_one_pre  = salesman_route[city_one_swap - 1];
    unsigned int trip_city_one_post = salesman_route[city_one_swap + 1];
    
    unsigned int trip_city_two      = salesman_route[city_two_swap];
    unsigned int trip_city_two_pre  = salesman_route[city_two_swap - 1];
    unsigned int trip_city_two_post = salesman_route[city_two_swap + 1];
    // The original and post distances
    float original_dist = 0;
    float proposal_dist = 0;
    
    // We will always have 4 calculations for original distance and the proposed distance, so we just unroll the loop here
    // TODO: It may be nice to make vars for the locations as well so this does not look so gross
    // The first city, unswapped. The one behind it and the one in front of it
    original_dist += (location[trip_city_one_pre].x - location[trip_city_one].x) *
                        (location[trip_city_one_pre].x - location[trip_city_one].x) +
                        (location[trip_city_one_pre].y - location[trip_city_one].y) *
                        (location[trip_city_one_pre].y - location[trip_city_one].y);
    original_dist += (location[trip_city_one_post].x - location[trip_city_one].x) *
                        (location[trip_city_one_post].x - location[trip_city_one].x) +
                        (location[trip_city_one_post].y - location[trip_city_one].y) *
                        (location[trip_city_one_post].y - location[trip_city_one].y);
    // The second city, unswapped. The one behind it and the one in front of it
    original_dist += (location[trip_city_two_pre].x - location[trip_city_two].x) *
                        (location[trip_city_two_pre].x - location[trip_city_two].x) +
                        (location[trip_city_two_pre].y - location[trip_city_two].y) *
                        (location[trip_city_two_pre].y - location[trip_city_two].y);
    original_dist += (location[trip_city_two_post].x - location[trip_city_two].x) *
                        (location[trip_city_two_post].x - location[trip_city_two].x) +
                        (location[trip_city_two_post].y - location[trip_city_two].y) *
                        (location[trip_city_two_post].y - location[trip_city_two].y);
    // The first city, swapped. The one behind it and the one in front of it
    proposal_dist += (location[trip_city_two_pre].x - location[trip_city_one].x) *
                        (location[trip_city_two_pre].x - location[trip_city_one].x) +
                        (location[trip_city_two_pre].y - location[trip_city_one].y) *
                        (location[trip_city_two_pre].y - location[trip_city_one].y);
    proposal_dist += (location[trip_city_two_post].x - location[trip_city_one].x) *
                        (location[trip_city_two_post].x - location[trip_city_one].x) +
                        (location[trip_city_two_post].y - location[trip_city_one].y) *
                        (location[trip_city_two_post].y - location[trip_city_one].y);
    // The second city, swapped. The one behind it and the one in front of it
    proposal_dist += (location[trip_city_one_pre].x - location[trip_city_two].x) *
                        (location[trip_city_one_pre].x - location[trip_city_two].x) +
                        (location[trip_city_one_pre].y - location[trip_city_two].y) *
                        (location[trip_city_one_pre].y - location[trip_city_two].y);
    proposal_dist += (location[trip_city_one_post].x - location[trip_city_two].x) *
                        (location[trip_city_one_post].x - location[trip_city_two].x) +
                        (location[trip_city_one_post].y - location[trip_city_two].y) *
                        (location[trip_city_one_post].y - location[trip_city_two].y);
    
    
            
    if (proposal_dist < original_dist){
      flag[tid] = 1;
      global_flag[0] = 1;
    } else {
      delta = proposal_dist - original_dist;
      p = exp(-delta/ T[0]);
      if (p > r[tid]){
        flag[tid] = 1;
        global_flag[0] = 1;
      }
    }
 }
 
                           
#endif // _TSP_SOLVE_H_
