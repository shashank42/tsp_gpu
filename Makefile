
tsp_cuda: main_solver.cu
	nvcc --optimize=5 --use_fast_math -arch=compute_35 main_solver.cu -o columbus -lcurand

clean:
	rm -f tsp_cuda
	

