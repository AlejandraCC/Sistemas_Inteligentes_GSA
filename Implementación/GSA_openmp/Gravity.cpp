// Gravity.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include "Gravitational.h"
#define PI 3.14159
#include <fstream>

float Ackley(std::vector<float> X)
{
	float a = 20.0f, b = 0.2f, c = 2 * (float)PI,  sum = 0.0f, cosenos = 0.0f;
	unsigned int d = X.size();
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
	{
		sum += 418.9829f - x * sin(sqrt(abs(x)));
	}
	return sum;
}

float Funcion_3(std::vector<float> X)
{
	float sum = 0.0f;
	for (auto x:X)
	{
		sum += x * x;
	}
	return 0.5 - (pow(sin(sqrt(sum)), 2) - 0.5) / pow(1.0 + 0.001*sum, 2);
}


int main()
{
	
	//Gravitational G1(Ackley, 30, 2, -32.768f, 32.768f, true);
	//Gravitational G1(Schwefel, 30, 2, -500.0f, 500.0f, true);
	
	
	Gravitational *G_100[100];
	FILE* datos;
	fopen_s(&datos, "datos.txt", "w");
	
#pragma omp parallel for
	for (int i = 0; i < 100; i++)
	{
		//G_100[i] = new Gravitational(Ackley, 60, 8, -32.768f, 32.768f, true);
		//G_100[i] = new Gravitational(Schwefel, 1200, 8, -500.0f, 500.0f, true);
		//G=16000 P=400
		//G_100[i] = new Gravitational(Funcion_3, 200, 8, -100.0f, 100.0f, false);
		G_100[i] = new Gravitational(Ackley, 60, 2, -32.768f, 32.768f, true);
		//G_100[i] = new Gravitational(Schwefel, 400, 2, -500.0f, 500.0f, true);
		//G=5000
		//G_100[i] = new Gravitational(Funcion_3, 200, 2, -100.0f, 100.0f, false);
		//G=60
		G_100[i]->optimizar(50);
	}
	
	for (size_t i = 0; i < 100; i++)
	{
		for (size_t j = 0; j < 50; j++)
		{
			fprintf_s(datos, "%f\t", G_100[i]->best_data[j]);
		}
		fprintf_s(datos, "\n");
	}
	fclose(datos);
	return 0;
}

