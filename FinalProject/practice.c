#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

struct Material {
    float ambient[3];
    float diffuse[3];
    float specular[3];
    float shininess;

};

/******************************************************
 *                  Function Prototypes
 *****************************************************/
void generateMaterial(struct Material* m);
void printMaterial(struct Material m);
double* normalize(double vec[]);
double dotProduct(double vec1[], double vec2[]);

int main()
{
    srand ( time(NULL) );

    struct Material m; // Material
    double V[] = {8, 1, -2}; //direction pointing towards the viewer. Assume origin
    double L[] = {1, 2, 4}; //direction from point on the surface toward each light source
    double* N; //Normal at point P
    int R[3]; //Direction of reflection. Is it given????????

    return 0;
}



/**
 * Generate rgb values for each light component
 */
void generateMaterial(struct Material* m)
{
    for (int i=0; i<3; i++)
    {
        m -> ambient[i] = (rand() % 101) / 100.0f;
        m -> diffuse[i] = (rand() % 101) / 100.0f;
        m -> specular[i] = (rand() % 101) / 100.0f;
    }

    //Assume shininess is always 1
    m -> shininess = 1;
}

/**
 * THis function will not be used. It is just for
 * debugging purposes
 */
void printMaterial(struct Material m)
{
    for (int i=0; i<3; i++)
    {
        printf("%.2f ", m.ambient[i]);
    }

    printf("\n");

    for (int i=0; i<3; i++)
    {
        printf("%.2f ", m.diffuse[i]);
    }

    printf("\n");

    for (int i=0; i<3; i++)
    {
        printf("%.2f ", m.specular[i]);
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

