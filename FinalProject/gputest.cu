#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

/////////////////////////////////////////////////////////////////////////////////////////////////////
//                              FUNCTIONS IMPLEMENTATION                                           //
/////////////////////////////////////////////////////////////////////////////////////////////////////

struct Material {
  unsigned char ambient[3]; 
  unsigned char diffuse[3];
  unsigned char specular[3];
  double shininess;

};

struct Light{
  int xcord;
  int ycord;
  int zcord;
  unsigned char Spec[3];
  unsigned char Amb[3];
  unsigned char Diff[3];
};

/**
 * Generate rgb values for each light component
 */
 void generateMaterial(struct Material* m)
 {
     int i;
     for (i=0; i<3; i++)
     {
         m -> ambient[i] = (rand() % 256);
         m -> diffuse[i] = (rand() % 256);
         m -> specular[i] = (rand() % 256);
     }
 
     //Assume shininess is always 1
     m -> shininess = 1;
 }
 
/**
 * Generate lights
 */
 void generateLights(struct Light* l, int numRows, int numCols)
 {
     int i;
     l->xcord = (rand()% numRows);
     l->ycord = (rand()% numCols);
     l->zcord = 0;

     for (i=0; i<3; i++)
     {
         l -> Amb[i] = (rand() % 256);
         l -> Spec[i] = (rand() % 256);
         l -> Diff[i] = (rand() % 256);
     }

 }

 /**
  * THis function will not be used. It is just for
  * debugging purposes
  */
 void printMaterial(struct Material m)
 {
     int i, j, k;
     for ( i=0; i<3; i++)
     {
         printf("%u ", m.ambient[i]);
     }
 
     printf("\n");
 
     for ( j=0; j<3; j++)
     {
         printf("%u ", m.diffuse[j]);
     }
 
     printf("\n");
 
     for ( k=0; k<3; k++)
     {
         printf("%u ", m.specular[k]);
     }
 
     printf("\n");
 }
 
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
 
     int h;
     for ( h=0; h<3; h++)
     {
         SquaredTotal += pow(vec[h], 2);
     }
 
     magnitude = sqrt(SquaredTotal);
 
     //normalize the vector
     int j;
     for (j=0; j<3; j++)
     {
         tempN[j] = vec[j] / magnitude;
     }
 
     return tempN;
 }
 
 /**
  * Return the dot product between two vectors
  */
 double dotProduct(double vec1[], double vec2[])
 {
     double result = 0;
 
     int h;
     for (h=0; h<3; h++)
     {
         result += vec1[h] * vec2[h];
     }
 
     return result;
 }
 
 
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

// GPU function to square root values
__global__ void gpu_sqrt(float* a, long N) {
  long element = blockIdx.x*blockDim.x + threadIdx.x;
  if (element < N) a[element] = sqrt(a[element]);
}


//Create three different arrays of channels
__global__ void separateChannels(const uchar4* const d_original,
                                unsigned char* const redChannel,
                                unsigned char* const greenChannel,
                                unsigned char* const blueChannel,
                                int numRows, int numCols) {

  const int2 indexTuple = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                    blockIdx.y * blockDim.y + threadIdx.y);
  

  if (indexTuple.x < numCols &&  indexTuple.y < numRows) {
    // pixel index
    const int index = indexTuple.y * numCols + indexTuple.x;

    redChannel[index]   = d_original[index].x;
    greenChannel[index] = d_original[index].y;
    blueChannel[index]  = d_original[index].z;

    // printf("%u\n", d_original[11957].x);
  }

}


/**
* @author: Arelys Alvarez
* @params d_output Final image
* @params redChannel The R component of the image
* @params greenChannel The G component of the image
* @params blueChannel The B component of the image
* @params numRows Number of rows in image
* @params numCols Number of columns in image
*
* This kernel takes in three color channels and recombines them
* into one image.  The alpha channel is set to 255 to represent
* that this image has no transparency.
**/
__global__
void recombineChannels(uchar4* const d_output,
                       const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       int numRows,
                       int numCols){

  const int2 indexTuple = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                    blockIdx.y * blockDim.y + threadIdx.y);


  // Check if threads are within the boundaries of the image
  if (indexTuple.x < numCols &&  indexTuple.y < numRows) {

    // pixel index
    const int index = indexTuple.y * numCols + indexTuple.x;

    //Alpha should be 255 for no transparency
    uchar4 outputPixel = make_uchar4(redChannel[index], greenChannel[index], blueChannel[index], 255);

    d_output[index] = outputPixel;

  }

}


// GPU function for the Phong reflection model
__global__ void gpu_phong(const uchar4* d_original,
                          const unsigned char* inputChannel, char channelFlag, //we need the original picture here
                          unsigned char* outputChannel,
                          const Light* lightSource, Material* material,
                          int numRows, int numCols, int numLights) {
  
  float3 viewVector, lightVector, reflectionVector;

  // unsigned char surfacePoint = blockIdx.x * blockDim.x + threadIdx.x;

  const int2 indexTuple = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                    blockIdx.y * blockDim.y + threadIdx.y);

  if (indexTuple.x < numCols &&  indexTuple.y < numRows) {
  
    // pixel index
  const int index = indexTuple.y * numCols + indexTuple.x;

  float3 point = make_float3(d_original[index].x, d_original[index].y, d_original[index].z);
  viewVector = make_float3(5.0f, 5.0f, 5.0f); // Assume this view position

  // norm3d returns the squared root of the squared sum of its parameters
  double pointVectorMagnitude = norm3d(point.x, point.y, point.z);
  double viewMagnitude = norm3d(viewVector.x, viewVector.y, viewVector.z);

  // normalized point
  point.x = point.x /pointVectorMagnitude;
  point.y = point.y /pointVectorMagnitude;
  point.z = point.z /pointVectorMagnitude;

  //normalized view vector
  viewVector.x = viewVector.x / viewMagnitude;
  viewVector.y = viewVector.y / viewMagnitude;
  viewVector.z = viewVector.z / viewMagnitude;

  int i;

  for (i=0; i<numLights; i++) {
    lightVector = make_float3(lightSource[i].xcord - point.x,
                              lightSource[i].ycord - point.y,
                              lightSource[i].zcord - point.z);

    double lightVectorMagnitude = norm3d(lightVector.x, lightVector.y, lightVector.z);

    // normalized light vector
    lightVector.x = lightVector.x / lightVectorMagnitude;
    lightVector.y = lightVector.y / lightVectorMagnitude;
    lightVector.z = lightVector.z / lightVectorMagnitude;


    //dot product of normalized lightVector x normalized point
    double NdotL = (point.x * lightVector.x) + (point.y * lightVector.y) + (point.z * lightVector.z);

    reflectionVector = make_float3(2 * NdotL * point.x - lightVector.x,
                                  2 * NdotL * point.y - lightVector.y,
                                  2 * NdotL * point.z - lightVector.z);

    // magnitude of the reflection vector
    double reflectionVectorMagnitude = norm3d(point.x, point.y, point.z);

    // normalize the reflection vector
    reflectionVector.x = reflectionVector.x / reflectionVectorMagnitude;
    reflectionVector.y = reflectionVector.y / reflectionVectorMagnitude;
    reflectionVector.z = reflectionVector.z / reflectionVectorMagnitude;

    double RdotV = (reflectionVector.x * viewVector.x) +
                    (reflectionVector.y * viewVector.y) +
                    (reflectionVector.z * viewVector.z);

    double ambientTerm;
    double diffuseTerm;
    double specularTerm;


    // if channel is red
    if(channelFlag == 'R'){

      ambientTerm = material->ambient[0] * lightSource[i].Amb[0];
      diffuseTerm = material->diffuse[0] * fmax( NdotL, 0.0) * lightSource[i].Diff[0];
      specularTerm = material->specular[0] * pow(fmax(RdotV, 0.0), material->shininess) * lightSource[i].Spec[0];

    }
    
    // if channel is green
    else if (channelFlag == 'G'){
      ambientTerm = material->ambient[1] * lightSource[i].Amb[1];
      diffuseTerm = material->diffuse[1] * fmax( NdotL, 0.0) * lightSource[i].Diff[1];
      specularTerm = material->specular[1] * pow(fmax(RdotV, 0.0), material->shininess) * lightSource[i].Spec[1];
    }

    // if channel is blue
    else if (channelFlag == 'B'){
      ambientTerm = material->ambient[2] * lightSource[i].Amb[2];
      diffuseTerm = material->diffuse[2] * fmax( NdotL, 0.0) * lightSource[i].Diff[2];
      specularTerm = material->specular[2] * pow(fmax(RdotV, 0.0), material->shininess) * lightSource[i].Spec[2];
    }

    if (NdotL < 0) {
      diffuseTerm = 0;
      specularTerm = 0;
    } 

    if (RdotV < 0) {
      specularTerm = 0;
    }


    outputChannel[index] += (unsigned char) ambientTerm + (unsigned char) diffuseTerm + (unsigned char) specularTerm;

    

    
  }


  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

// Normal C function to square root values
void normal(float* a, long N)                                                                                                                                                                                     
{
  long i;                                                                                                                                                                                                                
  for (i = 0; i < N; ++i)                                                                                                                                                                                    
    a[i] = sqrt(a[i]);                                                                                                                                                                                           
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

  srand(time(0)); //to use srand funtion

  Mat image = imread("original.jpg", 1); //original image in BGR format
  Mat imageRGBA; //original image in RGBA format

  //convert an image from BGR channel to RGBA and store it in imageRGBA
  cvtColor(image, imageRGBA, CV_BGR2RGBA);


  uchar4 *h_original, *d_original, *combinedImageInput, *combinedImageOutput;
  unsigned char *h_output, *d_output; 
  unsigned char *d_red, *d_green, *d_blue, *red, *green, *blue, *d_redOutput, *d_greenOutput, *d_blueOutput ;
  Material *h_material, *d_material;
  struct Light h_lights[2], *d_lights;

  const size_t numPixels = image.rows * image.cols;

  h_original = (uchar4 *) imageRGBA.ptr<unsigned char>(0);

  combinedImageOutput = (uchar4*) malloc(numPixels * sizeof(uchar4));
  h_output = (unsigned char*) malloc(numPixels * sizeof(unsigned char));
  red = (unsigned char*) malloc(numPixels * sizeof(unsigned char));
  green = (unsigned char*) malloc(numPixels * sizeof(unsigned char));
  blue = (unsigned char*) malloc(numPixels * sizeof(unsigned char));
  h_material = (Material*) malloc(numPixels * sizeof(Material));

  generateMaterial(h_material); //generate random material

  int numLights = 2; //number of lights
  int k;
  for (k = 0; k < numLights; k++){
    srand(time(0));
    generateLights(&h_lights[k], image.rows, image.cols);
  }

  // Alocate memory space in GPU
  cudaMalloc(&d_original, sizeof(uchar4) * numPixels);
  cudaMalloc(&combinedImageInput, sizeof(uchar4) * numPixels);
  cudaMalloc(&d_output, sizeof(unsigned char) * numPixels);
  cudaMalloc(&d_red, sizeof(unsigned char) * numPixels);
  cudaMalloc(&d_green, sizeof(unsigned char) * numPixels);
  cudaMalloc(&d_blue, sizeof(unsigned char) * numPixels);
  cudaMalloc(&d_material, sizeof(Material));
  cudaMalloc(&d_lights, sizeof(Light) * 2);

  //copy original image to the gpu
  cudaMemcpy(d_original, h_original, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
  cudaMemcpy(combinedImageInput, h_original, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
  cudaMemcpy(d_material, h_material, sizeof(Material), cudaMemcpyHostToDevice);   //copy material to gpu
  cudaMemcpy(d_lights, h_lights, sizeof(Material) * 2, cudaMemcpyHostToDevice);
 
 

  //opencv inverts rows and cols
  int rows = image.rows;
  int cols = image.cols;
  const dim3 blockSize(32, 32);
  const dim3 gridSize(image.rows/32 + 1, image.cols/32 + 1);

  separateChannels<<<gridSize, blockSize>>>(d_original, d_red, d_green, d_blue, rows, cols);
  gpu_phong<<<gridSize, blockSize>>>(d_original, d_red, 'R', d_redOutput, d_lights, d_material, rows, cols, numLights);
  gpu_phong<<<gridSize, blockSize>>>(d_original, d_green, 'G', d_greenOutput, d_lights, d_material, rows, cols, numLights);
  gpu_phong<<<gridSize, blockSize>>>(d_original, d_blue, 'B', d_blueOutput, d_lights, d_material, rows, cols, numLights);
  recombineChannels<<<gridSize, blockSize>>>(combinedImageOutput, d_red, d_green, d_blue, rows, cols);
  
  
  cudaMemcpy(red, d_redOutput, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
  cudaMemcpy(green, d_greenOutput, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
  cudaMemcpy(blue, d_blueOutput, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
  cudaMemcpy(combinedImageOutput, combinedImageInput, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);
 


  cudaFree(d_original);
  cudaFree(d_output);
  cudaFree(d_red);
  cudaFree(d_green);
  cudaFree(d_blue);
  cudaFree(combinedImageInput);

  Mat output(rows, cols, CV_8UC4, (void*) combinedImageOutput);
  imwrite("output.jpg", output);
  



  return 0;
}

