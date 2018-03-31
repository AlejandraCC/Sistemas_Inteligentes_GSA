// Gravity.cpp: define el punto de entrada de la aplicación de consola.
//

// #include <stdafx>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "GSA.h"

using namespace std;


float Ackley(std::vector<float> X)
{
	float a = 20.0f, b = 0.2f, c = 2 * M_PI, d = X.size(), sum = 0.0f, cosenos = 0.0f;
	for (auto x:X)
	{
		sum += x * x;
		cosenos += cos(c*x);
	}
	return -a * exp(-b * sqrt(sum / d)) - exp(cosenos / d) + a + exp(1);
}

float Schwefel(std::vector<float> X) 
{
	float sum = 0.0f;
	for (auto x : X)
		sum +=  x * sin(sqrt(abs(x)));
	return 418.9829*X.size() - sum;
}

float Funcion_3(std::vector<float> X) 
{
	float sum = 0.0f;
	for (auto x:X)
        sum += x * x;

	return 0.5 - (   pow(sin(sqrt(sum)), 2) - 0.5   ) / pow(1.0 + 0.001*sum, 2);
}




int main()
{              
	//F2
	float G = 10000;
	int numero_poblacion = 600;
                                     
                                                                          
	int dimension = 2;
	bool minimizar = true;              

	for (int i = 0; i < 1; ++i)
	{
		float minimo = -32.768f, maximo = 32.768f;  
		Gravitational G1(Ackley, numero_poblacion, dimension, minimo, maximo, minimizar, G);
		// std::vector<float> x{0.0f, 0.0f};
		// std::cout<<"optimo: "<<Ackley(x)<<std::endl;
		G1.optimizar(50);         
	}                                       
	// bestPositionSoFar: 0.00329201 0.00169011 bestFitnessSoFar: 0.010831

          
	// for (int i = 0; i < 1; ++i)
	// {
	// 	float minimo = -500.0f, maximo = 500.0f;                                   
	// 	Gravitational G1(Schwefel, numero_poblacion, dimension, minimo, maximo, minimizar, G);
	// 	// std::vector<float> x{420.9687, 420.9687};
	// 	// std::cout<<"optimo: "<<Schwefel(x)<<std::endl;
	// 	G1.optimizar(50);
	// }
   // bestPositionSoFar: 420.985 420.882 bestFitnessSoFar: 0.00101729

           

	// for (int i = 0; i < 1; ++i)
	// {
	// 	minimizar = false;         
	// 	float minimo = -100.0f, maximo = 100.0f;  
	// 	Gravitational G1(Funcion_3, numero_poblacion, dimension, minimo, maximo, minimizar, G);
	// 	std::vector<float> x{0,0};
	// 	// std::cout<<"optimo: "<<Funcion_3(x)<<std::endl;
	// 	G1.optimizar(50);
	// }
	// bestPositionSoFar: 0.0131519 -0.0283799 bestFitnessSoFar: 0.999021

                                                    
	
	return 0;
}

                                                                          