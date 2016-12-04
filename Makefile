
tsp_cuda: main_solver.cu
	nvcc --optimize=5 --use_fast_math -arch=compute_35 -g -G main_solver.cu -o tsp_cuda -lcurand

clean:
	rm -f tsp_cuda
	

