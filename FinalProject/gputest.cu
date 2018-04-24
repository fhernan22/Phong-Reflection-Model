#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

long N = 6400000000;                                                                                                                                                                                         
int doPrint = 0; 

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// HELPER CODE TO INITIALIZE, PRINT AND TIME
struct timeval start, end;
void initialize(float *a, long N) {
  long i;
  for (i = 0; i < N; ++i) { 
    a[i] = pow(rand() % 10, 2); 
  }                                                                                                                                                                                       
}

void print(float* a, long N) {
   if (doPrint) {
   long i;
   for (i = 0; i < N; ++i)
      printf("%f ", a[i]);
   printf("\n");
   }
}  

void starttime() {
  gettimeofday( &start, 0 );
}

void endtime(const char* c) {
   gettimeofday( &end, 0 );
   double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
   printf("%s: %f ms\n", c, elapsed); 
}

void init(float* a, long N, const char* c) {
  printf("***************** %s **********************\n", c);
  // TMC Commenting Out for Class  
  //printf("Initializing array....\n");
  //initialize(a, N); 
  //printf("Done.\n");
  print(a, N);
  printf("Running %s...\n", c);
  starttime();
}

void finish(float* a, long N, const char* c) {
  endtime(c);
  printf("Done.\n");
  print(a, N);
  printf("***************************************************\n");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////



// Normal C function to square root values
void normal(float* a, long N)                                                                                                                                                                                     
{
  long i;                                                                                                                                                                                                                
  for (i = 0; i < N; ++i)                                                                                                                                                                                    
    a[i] = sqrt(a[i]);                                                                                                                                                                                           
}                 

// GPU function to square root values
__global__ void gpu_sqrt(float* a, long N) {
   long element = blockIdx.x*blockDim.x + threadIdx.x;
   if (element < N) a[element] = sqrt(a[element]);
}

void gpuFunc(float* a, long N) {
   int numThreads = 1024; // This can vary, up to 1024
   long numCores = N / 1024 + 1;

   float* gpuA;
   cudaMalloc(&gpuA, N*sizeof(float)); // Allocate enough memory on the GPU
   cudaMemcpy(gpuA, a, N*sizeof(float), cudaMemcpyHostToDevice); // Copy array from CPU to GPU
   gpu_sqrt<<<numCores, numThreads>>>(gpuA, N);  // Call GPU Sqrt
   cudaMemcpy(a, gpuA, N*sizeof(float), cudaMemcpyDeviceToHost); // Copy array from GPU to CPU
   cudaFree(&gpuA); // Free the memory on the GPU
}
                                                                                                                                                                                               
 

int main()                                                                                                                                                                                  
{                                                                                                                                                                                                                
  //////////////////////////////////////////////////////////////////////////
  // Necessary if you are doing SSE.  Align on a 128-bit boundary (16 bytes)
  /*float* a;                                                                                                                                                                                                      
  posix_memalign((void**)&a, 16,  N * sizeof(float));                                                                                                                                                            
  /////////////////////////////////////////////////////////////////////////

  // Test 1: Sequential For Loop
  init(a, N, "Normal");
  normal(a, N); 
  finish(a, N, "Normal"); 

  // Test 2: Vectorization
  init(a, N, "GPU");
  gpuFunc(a, N);  
  finish(a, N, "GPU");*/

  Mat image = imread("original.jpg", 1); //original image in BGR format
  Mat imageRGBA; //original image in RGBA format

  //convert an image from BGR channel to RGBA
  cvtColor(image, imageRGBA, CV_BGR2RGBA);


  uchar4 *h_original, *d_original;
  unsigned char *h_output, *d_output; 

  h_original = (uchar4 *)imageRGBA.ptr<unsigned char>(0);

  const size_t numPixels = image.rows * image.cols;

  // Alocate memory space in GPU
  cudaMalloc(&d_original, sizeof(uchar4) * numPixels);
  cudaMalloc(&d_output, sizeof(unsigned char) * numPixels);

  //copy original image to the gpu
  cudaMemcpy(&d_original, &h_original, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);


  //Launch kernel here...


  printf("rows: %d\n", image.rows);
  printf("cols: %d\n",image.cols);
  printf("pixels: %d\n", numPixels);

  printf("%u\n", h_original->x);
  printf("%u\n", h_original->y);
  printf("%u\n", h_original->z);
  printf("%u\n", h_original->w);

  return 0;
}

