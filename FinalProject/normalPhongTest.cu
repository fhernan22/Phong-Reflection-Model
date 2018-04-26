#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
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

struct Material 
{
    float ambient[3]; // ambient reflection constant for RGB respectively
    float diffuse[3]; // diffuse reflection constant for RGB respectively
    float specular[3]; // specular reflection constant for RGB respectively
    double shininess; // shininess constant
}

struct Lights // Light source coordinates
{
	int xcord;
	int ycord;
	int zcord;
	float iSpec[3]; // RGB intensity for specular component of light
	float iDiff[3]; // RGB intensity for diffuse component of light
	float iAmb[3]; // RGB intensity for ambient component of light
}

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

/**
 * Return a pointer to a normalized vector
 */
double* normalize(double vec[])
{
    double magnitude;

    double* tempN;

    //Allocate space in memory for tempN
    tempN = (double*) malloc(3 * sizeof(double));

    //store the sum of each component squared in vec 
    int SquaredTotal = 0; 

    for (int i=0; i<3; i++)
    {
        SquaredTotal += pow(vec[i], 2);
    }

    magnitude = sqrt(SquaredTotal);

    //normalize the vector
    for (int i=0; i<3; i++)
    {
        tempN[i] = vec[i] / magnitude;
    }

    return tempN;
}

/**
 * Return the dot product between two vectors
 */
double dotProduct(double vec1[], double vec2[])
{
    double result = 0;

    for (int i=0; i<3; i++)
    {
        result += vec1[i] * vec2[i];
    }

    return result;
}

void phongRed(Material m, Lights* light, double* V, double* N, long Num, int NumL, Mat img)
{
	for (i = 0; i < NumL; i++) // For each light source
	   {
		   for (x = 0; x < img.rows; x++;) // Iterate through rows of pixels
		   {
			   for (y = 0; y < img.cols; y++) // Iterate through columns of pixels
			   {
				   Vec3b intensity = img.at<Vec3b>(y, x); // BGR intensity of pixel
				   
				   // Image is treated as a plane, i.e. all pixels share the same z coordinate
				   double L[] = {(light[i].xcord - x), (light[i].ycord - y), (light[i].zcord - 0}; // direction vector from point towards light source
				   L = normalize(L); // Normalize L
		           	   double R[3]; // direction that a perfectly reflected ray of light would take, given by R = 2(L dot N)N - L
		                   R = {(2*dotProduct(N, L)*N[0] - L[0]), (2*dotProduct(N, L)*N[1] - L[1]), (2*dotProduct(N, L)*N[2] - L[2])};
		                   R = normalize(R); // Normalize R 
				   
				   // Material struct stores constants in RGB order, but it doesn't really matter
				   tAmb = m.ambient[0]*light[i].iAmb[0]; // Ambient term
		                   tDiff = m.diffuse[0]*dotProduct(N, L)*light[i].iDiff[0]; // Diffuse term
		                   tSpec = m.specular[0]*pow(dotProduct(R, V), m.shininess)*light[i].iSpec[0]; // Specular term
				   
				   if (dotProduct(p[element].N, L) < 0) // Only include a term if its dot product is positive
		           {
						tDiff = 0;
						tSpec = 0; // Additionally, the specular term should only be included if the dot product of the diffuse term is positive
				   }
		   
				   if (dotProduct(R, V) < 0) // Only include a term if its dot product is positive
				   {
						tSpec = 0;
				   }
				   
				   intensity.val[2] += tAmb+tDiff+tSpec; // Change red intensity value
				   img.at<Vec3b>(y, x) = intensity; // Change pixel
			   }
		   }
	   }
}

void phongGreen(Material m, Lights* light, double* V, double* N, long Num, int NumL, Mat img)
{
	for (i = 0; i < NumL; i++) // For each light source
	   {
		   for (x = 0; x < img.rows; x++;) // Iterate through rows of pixels
		   {
			   for (y = 0; y < img.cols; y++) // Iterate through columns of pixels
			   {
				   Vec3b intensity = img.at<Vec3b>(y, x); // BGR intensity of pixel
				   
				   // Image is treated as a plane, i.e. all pixels share the same z coordinate
				   double L[] = {(light[i].xcord - x), (light[i].ycord - y), (light[i].zcord - 0}; // direction vector from point towards light source
				   L = normalize(L); // Normalize L
		                   double R[3]; // direction that a perfectly reflected ray of light would take, given by R = 2(L dot N)N - L
		                   R = {(2*dotProduct(N, L)*N[0] - L[0]), (2*dotProduct(N, L)*N[1] - L[1]), (2*dotProduct(N, L)*N[2] - L[2])};
		                   R = normalize(R); // Normalize R 
				   
				   // Material struct stores constants in RGB order, but it doesn't really matter
				   tAmb = m.ambient[1]*light[i].iAmb[1]; // Ambient term
		                   tDiff = m.diffuse[1]*dotProduct(N, L)*light[i].iDiff[1]; // Diffuse term
		                   tSpec = m.specular[1]*pow(dotProduct(R, V), m.shininess)*light[i].iSpec[1]; // Specular term
		 		   
				   if (dotProduct(p[element].N, L) < 0) // Only include a term if its dot product is positive
		           {
						tDiff = 0;
						tSpec = 0; // Additionally, the specular term should only be included if the dot product of the diffuse term is positive
				   }
		   
				   if (dotProduct(R, V) < 0) // Only include a term if its dot product is positive
				   {
						tSpec = 0;
				   }
				   
				   intensity.val[1] += tAmb+tDiff+tSpec; // Change green intensity value
				   img.at<Vec3b>(y, x) = intensity; // Change pixel
			   }
		   }
	   }
}

void phongBlue(Material m, Lights* light, double* V, double* N, long Num, int NumL, Mat img)
{
	for (i = 0; i < NumL; i++) // For each light source
	   {
		   for (x = 0; x < img.rows; x++;) // Iterate through rows of pixels
		   {
			   for (y = 0; y < img.cols; y++) // Iterate through columns of pixels
			   {
				   Vec3b intensity = img.at<Vec3b>(y, x, z); // BGR intensity of pixel
				   
				   // Image is treated as a plane, i.e. all pixels share the same z coordinate
				   double L[] = {(light[i].xcord - x), (light[i].ycord - y), (light[i].zcord - 0}; // direction vector from point towards light source
				   L = normalize(L); // Normalize L
		                   double R[3]; // direction that a perfectly reflected ray of light would take, given by R = 2(L dot N)N - L
		                   R = {(2*dotProduct(N, L)*N[0] - L[0]), (2*dotProduct(N, L)*N[1] - L[1]), (2*dotProduct(N, L)*N[2] - L[2])};
		                   R = normalize(R); // Normalize R 
				   
				   // Material struct stores constants in RGB order, but it doesn't really matter
				   tAmb = m.ambient[2]*light[i].iAmb[2]; // Ambient term
		                   tDiff = m.diffuse[2]*dotProduct(N, L)*light[i].iDiff[2]; // Diffuse term
		                   tSpec = m.specular[2]*pow(dotProduct(R, V), m.shininess)*light[i].iSpec[2]; // Specular term
				   
				   if (dotProduct(p[element].N, L) < 0) // Only include a term if its dot product is positive
		           {
						tDiff = 0;
						tSpec = 0; // Additionally, the specular term should only be included if the dot product of the diffuse term is positive
				   }
		   
				   if (dotProduct(R, V) < 0) // Only include a term if its dot product is positive
				   {
						tSpec = 0;
				   }
				   
				   intensity.val[0] += tAmb+tDiff+tSpec; // Change blue intensity value
				   img.at<Vec3b>(y, x) = intensity; // Change pixel
			   }
		   }
	   }
}

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

  init(0, 0, "Normal");
	
  Mat image = imread("original.jpg", 1); //original image in BGR format
  Mat imageRGBA; //original image in RGBA format
  
  double V[] = {7, 7, 7}; // normalized direction vector pointing towards viewer or camera, this is the same for all points
  double N[] = {0, 1, 0}; // normalized normal vector pointing up, this is the same for all pixels on the plane

  Mat img2 = image; // Copy of original image

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

  finish(0, 0, "Normal");	
  return 0;
}
