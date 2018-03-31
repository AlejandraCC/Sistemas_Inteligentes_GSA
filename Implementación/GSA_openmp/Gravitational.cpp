#include "stdafx.h"
#include "Gravitational.h"
bool agentemayor(Agente i, Agente j) { return (i.fitness < j.fitness); }

Gravitational::Gravitational(float(*funcion)(std::vector<float>), unsigned int numero_poblacion, unsigned int dimension, float minimo, float maximo, bool minimizar)
{
	//(a)Search space identification.
	this->dimension = dimension;
	this->funcion = funcion;
	this->max = maximo;
	this->min = minimo;
	this->minimizacion = minimizar;
	//(b)Randomized initialization.
	std::random_device generador;
	std::uniform_real_distribution<float> distribucion(min, max);
	for (size_t i = 0; i < numero_poblacion; i++)
	{
		poblacion.push_back(Agente(dimension, generador, distribucion));
	}
}


Gravitational::~Gravitational()
{
}


// optimizacion
float Gravitational::optimizar(int iteraciones)
{
	int indice = 0;
	float G = 16.0f;
	float G_step = G / iteraciones;
	float best, worst;
	std::random_device generador;
	std::uniform_real_distribution<float> distribucion(0,1);
	float resultado = 0.0f;
	do
	{
		//(c)Fitness evaluation of agents.
		for (auto &agent : poblacion)
		{
			agent.fitness = funcion(agent.data);
		}
		//(d)Update G(t), best(t), worst(t) and Mi(t) for i = 1, 2, ..., N.
		G-=G_step;
		std::sort(poblacion.begin(), poblacion.end(), agentemayor);
		if (minimizacion)
		{
			best = poblacion.at(0).fitness;
			worst = poblacion.at(poblacion.size() - 1).fitness;
		}
		else
		{
			worst = poblacion.at(0).fitness;
			best = poblacion.at(poblacion.size() - 1).fitness;
		}
		best_data[indice%50] = best;
		indice++;
		float sum = 0.0f;
		for (auto &agent : poblacion)
		{
			agent.m = (agent.fitness - worst) / (best - worst);
			sum += agent.m;
		}
		for (auto &agent : poblacion)
		{
			agent.Ma = agent.Mp = agent.Mi = agent.m / sum;
		}
		//(e)Calculation of the total force in different directions.
		int l = std::min<int>(iteraciones, poblacion.size());
		for (size_t i = 0; i < poblacion.size(); i++)
		{
			//vaciamos forces
			for (size_t k = 0; k < dimension; k++)
			{
				poblacion[i].forces[k] = 0;
			}
			//for (size_t j = 0; j < poblacion.size(); j++)
			//mejora, solo los mejoras van a influir.
			for (size_t j = 0; j < l; j++)
			{
				if (i != j)
				{
					auto Force = F(poblacion[i], poblacion[j]);
					float randj = distribucion(generador);
					for (size_t k = 0; k < dimension; k++)
					{
						poblacion[i].forces[k] += randj*G*Force[k];
					}
				}
			}
		}
		//(f)Calculation of acceleration and velocity.
		for (auto &agent:poblacion)
		{
			for (size_t i = 0; i < dimension; i++)
			{
				if (agent.Mi == 0)
					agent.acc[i] = 0;
				else
					agent.acc[i] = agent.forces[i] / agent.Mi;
			}
		}
		for (auto &agent : poblacion)
		{
			float randi = distribucion(generador);
			for (size_t i = 0; i < dimension; i++)
			{
				agent.vel[i] = randi * agent.vel[i] + agent.acc[i];
			}
		}
		//(g)Updating agents’ position.
		for (auto &agent : poblacion)
		{
			for (size_t i = 0; i < dimension; i++)
			{
				agent.data[i] = std::min<float>(agent.data[i] + agent.vel[i], this->max);
				agent.data[i] = std::max<float>(agent.data[i], this->min);
			}
		}
		//(h)Repeat steps c to g until the stop criteria is reached.
	} while (--iteraciones);
	
	//(i)End.
	return poblacion[0].fitness;
}


std::vector<float> Gravitational::F(Agente i, Agente j)
{
	float R = 0.0f;
	int dim = i.data.size();
	for (size_t pos = 0; pos < dim; pos++)
	{
		R += pow(i.data[pos] - j.data[pos],2);
	}
	R = sqrt(R);
	std::vector<float> Force;
	float e = 6.2204e-08f;//algun valor pequeño
	float sub_F = i.Mp*j.Ma / (R + e);
	for (size_t pos = 0; pos < dim; pos++)
	{
		Force.push_back(sub_F*(j.data[pos] - i.data[pos]));
	}
	return Force;
}
