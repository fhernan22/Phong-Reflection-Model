# GPU
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

gpu: gputest.cu
	/usr/local/cuda-9.1/bin/nvcc gputest.cu -o gputest $(LIBS)
