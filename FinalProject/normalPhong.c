//

struct Material 
{
    float ambient[3]; // ambient reflection constant for RGB respectively
    float diffuse[3]; // diffuse reflection constant for RGB respectively
    float specular[3]; // specular reflection constant for RGB respectively
    double shininess; // shininess constant
};

struct Lights // Light source coordinates
{
	int xcord;
	int ycord;
	int zcord;
	float iSpec[3]; // RGB intensity for specular component of light
	float iDiff[3]; // RGB intensity for diffuse component of light
	float iAmb[3]; // RGB intensity for ambient component of light
};

double V[] = {7, 7, 7}; // normalized direction vector pointing towards viewer or camera, this is the same for all points
double N[] = {0, 1, 0}; // normalized normal vector pointing up, this is the same for all pixels on the plane

Mat image = imread("original.jpg", 1); //original image in BGR format
Mat imageRGBA; //original image in RGBA format

Mat img2 = image; // Copy of original image

// Function to calculate lighting using phong model
// Parameters: Material of object, lights, vector V, normal vector N, number Num of pixels, and number NumL of lights
// All points share the same material and V
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
				   Vec3b intensity = img.at<Vec3b>(y, x); // BGR intensity of pixel
				   
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
