/*
    columbus: Software for computing approximate solutions to the traveling
   salesman's problem on GPUs Copyright (C) 2016 Steve Bronder and Haoyan Min

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

#ifndef _TSP_RNG_H_
#define _TSP_RNG_H_

#include <curand.h>
#include <curand_kernel.h>

/* this GPU kernel function is used to initialize the random states
 *  Come from:
 *    http://cs.umw.edu/~finlayson/class/fall14/cpsc425/notes/23-cuda-random.html
 *
 */
__global__ void init(unsigned int seed, curandState_t *states) {

  /* we have to initialize the state */
  /* the seed can be the same for each core, here we pass the time in from the
   * CPU */
  /* the sequence number should be different for each core (unless you want all
     cores to get the same sequence of numbers for some reason - use thread id!
   */
  /* the offset is how much extra we advance in the sequence for each call, can
   * be 0 */
  curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, 0,
              &states[blockIdx.x * blockDim.x + threadIdx.x]);
}

#endif // _TSP_RNG_H_
