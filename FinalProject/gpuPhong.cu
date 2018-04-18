//

struct Point // Points on surface of object
{
	float rgb[3]; // rgb light values of surface point respectively
	int xcord;
	int ycord;
	int zcord;
	double N[3];  // normalized vector for the normal at this point, must be given
}

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

double V[] = {7, 7, 7}; // normalized direction vector pointing towards viewer or camera, this is the same for all points

// GPU function to calculate lighting using phong model
// Parameters: Material of object, points on surface of object, lights, vector V, number Num of points, and number NumL of lights
// All points share the same material and V
__global__ void gpu_phongRed(Material m, Point* p, Lights* light, double* V, long Num, int NumL)
{
   long element = blockIdx.x*blockDim.x + threadIdx.x;
   if (element < Num)
   {	   
	   for (i = 0; i < NumL; i++) // For each light source
	   {
		   double L[] = {(light[i].xcord - p[element].xcord), (light[i].ycord - p[element].ycord), (light[i].zcord - p[element].zcord)}; // direction vector from point towards light source
		   L = normalize(L); // Normalize L
		   double R[3]; // direction that a perfectly reflected ray of light would take, given by R = 2(L dot N)N - L
		   R = {(2*dotProduct(p[element].N, L)*p[element].N[0] - L[0]), (2*dotProduct(p[element].N, L)*p[element].N[1] - L[1]), (2*dotProduct(p[element].N, L)*p[element].N[2] - L[2])};
		   R = normalize(R); // Normalize R
		   
		   tAmb = m.ambient[0]*light[i].iAmb[0]; // Ambient term
		   tDiff = m.diffuse[0]*dotProduct(p[element].N, L)*light[i].iDiff[0]; // Diffuse term
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
		   
		   p[element].rgb[0] += tAmb+tDiff+tSpec;
	   }
   }
}

__global__ void gpu_phongGreen(Material m, Point* p, Lights* light, double* V, long Num, int NumL)
{
   long element = blockIdx.x*blockDim.x + threadIdx.x;
   if (element < Num)
   {	   
	   for (i = 0; i < NumL; i++) // For each light source
	   {
		   double L[] = {(light[i].xcord - p[element].xcord), (light[i].ycord - p[element].ycord), (light[i].zcord - p[element].zcord)}; // direction vector from point towards light source
		   L = normalize(L); // Normalize L
		   double R[3]; // direction that a perfectly reflected ray of light would take, given by R = 2(L dot N)N - L
		   R = {(2*dotProduct(p[element].N, L)*p[element].N[0] - L[0]), (2*dotProduct(p[element].N, L)*p[element].N[1] - L[1]), (2*dotProduct(p[element].N, L)*p[element].N[2] - L[2])};
		   R = normalize(R); // Normalize R
		   
		   tAmb = m.ambient[1]*light[i].iAmb[1]; // Ambient term
		   tDiff = m.diffuse[1]*dotProduct(p[element].N, L)*light[i].iDiff[1]; // Diffuse term
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
		   
		   p[element].rgb[1] += tAmb+tDiff+tSpec;
	   }
   }
}

__global__ void gpu_phongBlue(Material m, Point* p, Lights* light, double* V, long Num, int NumL)
{
   long element = blockIdx.x*blockDim.x + threadIdx.x;
   if (element < Num)
   {	   
	   for (i = 0; i < NumL; i++) // For each light source
	   {
		   double L[] = {(light[i].xcord - p[element].xcord), (light[i].ycord - p[element].ycord), (light[i].zcord - p[element].zcord)}; // direction vector from point towards light source
		   L = normalize(L); // Normalize L
		   double R[3]; // direction that a perfectly reflected ray of light would take, given by R = 2(L dot N)N - L
		   R = {(2*dotProduct(p[element].N, L)*p[element].N[0] - L[0]), (2*dotProduct(p[element].N, L)*p[element].N[1] - L[1]), (2*dotProduct(p[element].N, L)*p[element].N[2] - L[2])};
		   R = normalize(R); // Normalize R
		   
		   tAmb = m.ambient[2]*light[i].iAmb[2]; // Ambient term
		   tDiff = m.diffuse[2]*dotProduct(p[element].N, L)*light[i].iDiff[2]; // Diffuse term
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
		   
		   p[element].rgb[2] += tAmb+tDiff+tSpec;
	   }
   }
}


