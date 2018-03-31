#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;


struct Agente  //particle
{
	std::vector<float> position, forces,velocity,acc;
	float Ma, Mp, Mi, m;
	float fitness; //personal best

	Agente(int dim,std::random_device &generador,std::uniform_real_distribution<float> &distribucion);
	
};



class Gravitational
{
private:
	std::vector<Agente> poblacion;
	unsigned int dimension;
	float min, max;
	bool minimizacion;
	float(*funcion)(std::vector<float>);
	cv::Mat heatMap;
	Mat heatMapWithPositions;

	float bestSoFarFitness;
	std::vector<float> bestSoFarPosition;

	float G;


	


public:
	Gravitational(float(*funcion)(std::vector<float>), unsigned int numero_poblacion, unsigned int dimension, float minimo, float maximo, bool minimizar, float G);
	~Gravitational();

	// optimizacion
	float optimizar(int iteraciones);
	void drawHeatMap();

	void initializeHeatMap();
	void drawPositionsInHeatMap();
	void drawPositionInHeatMap(std::vector<float> position);

	std::vector<float> F(Agente i, Agente j);
};







//************ GSA.cpp



bool agentemayorCmp(Agente i, Agente j) { return (i.fitness < j.fitness); }



Agente::Agente(int dim,std::random_device &generador,std::uniform_real_distribution<float> &distribucion)
	{
		for (size_t i = 0; i < dim; i++)
		{
			position.push_back(distribucion(generador));
			forces.push_back(0.0f);
			velocity.push_back(0.0f);
			acc.push_back(0.0f);
		}
		m=Ma = Mp = Mi = fitness = -1.0f;
	}

Gravitational::Gravitational(float(*funcion)(std::vector<float>), unsigned int numero_poblacion, unsigned int dimension, float minimo, float maximo, bool minimizar, float G)
{
	//(a)Search space identification.
	this->dimension = dimension;
	this->funcion = funcion;
	this->min = minimo;
	this->max = maximo;
	this->minimizacion = minimizar;

	this->G = G;

	if (minimizacion) bestSoFarFitness = 1000000000000.0;
	else bestSoFarFitness = -1000000000000.0;


	//(b)Randomized initialization.
	std::random_device generador;
	std::uniform_real_distribution<float> distribucion(min, max);
	for (size_t i = 0; i < numero_poblacion; i++)
		poblacion.push_back(Agente(dimension, generador, distribucion));
}


Gravitational::~Gravitational()
{
}

float Gconstant(int iteration,int max_it){
  float alfa=20, G0=100;
  return G0*exp(-alfa*iteration/max_it);   // equation 28
} 


void Gravitational::initializeHeatMap(){
	float zmin = 1000000000, zmax =-1000000000;
	Mat img = Mat::zeros(659, 650, CV_32F);
	float scaleFactor = img.cols/max;

	for ( int y=0; y<img.rows; y++ )
		for ( int x=0; x<img.cols; x++ )
		{
			//para que este centrado
		    std::vector<float> point{float(x-img.cols/2), float(y- img.rows/2)};  

            //escalar en el rango [min, max]
		    point[0] /= scaleFactor;
		    point[1] /= scaleFactor;

		    point[0] *= 2;
			point[1] *= 2;

		    float z = funcion(point);
		    img.at<float>(img.rows -1 - y, x) = z;

		    if (z < zmin)
		    	zmin = z;

		    if (z > zmax)
		    	zmax = z;
		}
	
	img.convertTo(heatMap, CV_8UC1, 255.0 / (zmax-zmin), -zmin); 
	applyColorMap(heatMap, heatMap, cv::COLORMAP_JET);

}

void Gravitational::drawPositionsInHeatMap(){


   // f1 = -2  , f3 = -5
	Mat heatMapWithPositions = heatMap.clone(); 
	float scaleFactor = heatMap.cols/(max*2);




	for (auto &agent : poblacion){
		float x = (agent.position[0] + max)*scaleFactor;
		float y = (agent.position[1] + max)*scaleFactor;
		circle(heatMapWithPositions, Point2f(x, heatMap.rows-5 - y), 3, Scalar(255, 255, 255),-1);
	}

	//draw bestSoFarPosition
	float x = (bestSoFarPosition[0] + max)*scaleFactor;
	float y = (bestSoFarPosition[1] + max)*scaleFactor;
	circle(heatMapWithPositions, Point2f(x, heatMap.rows-5 - y), 5, Scalar(0, 0, 0),-1);
	circle(heatMapWithPositions, Point2f(x, heatMap.rows-5 - y), 3, Scalar(0, 255, 255),-1);


	// x = ((0 + max)*scaleFactor);
	// y = ((0 + max)*scaleFactor);
	// circle(heatMapWithPositions, Point2f(x, heatMap.rows-5 - y), 2, Scalar(255, 0, 255),-1);

	
	imshow("heatMap", heatMapWithPositions);
	waitKey(0);
}



// optimizacion
float Gravitational::optimizar(int iteraciones)
{
	
  initializeHeatMap();

	float G_step = G / iteraciones;

	float best, worst;

	

	std::random_device generador;
	std::uniform_real_distribution<float> distribucion(0, 1);
	float resultado = 0.0f;

	// int max_iteraciones = iteraciones;
	// #pragma omp parallel for
	for (int iter = 0; iter < iteraciones; ++iter)
	{

		//************** %Checking allowable range. 


		//(c)Fitness evaluation of agents.
			for (auto &agent : poblacion)
				agent.fitness = funcion(agent.position);

			

		//(d)Update G(t), best(t), worst(t) and Mi(t) for i = 1, 2, ..., N.

            //***********ind min and max
			std::sort(poblacion.begin(), poblacion.end(), agentemayorCmp);
			if (minimizacion)
			{
				best  = poblacion.at(0).fitness;                        //Equation 17
				worst = poblacion.at(poblacion.size() - 1).fitness;     //Equation 18

				if(best < bestSoFarFitness) {
					bestSoFarFitness = best;
					bestSoFarPosition = poblacion.at(0).position;

					// drawPositionInHeatMap(poblacion.at(0).position);
				}
			}
			else
			{
				best = poblacion.at(poblacion.size() - 1).fitness;      //Equation 18
				worst = poblacion.at(0).fitness;                        //Equation 20

				if(best > bestSoFarFitness) {
					bestSoFarFitness = best;
					bestSoFarPosition = poblacion.at(poblacion.size() - 1).position;

					// drawPositionInHeatMap(bestSoFarPosition);
				}
			}

			drawPositionsInHeatMap();

		//*************keep track o the best so far


			float sum = 0.0f;
			for (auto &agent : poblacion)
			{
				agent.m = (agent.fitness - worst) / (best - worst); //Equation 15
				sum += agent.m;
			}

			for (auto &agent : poblacion)
				agent.Ma = agent.Mp = agent.Mi = agent.m / sum;     //Equation 16




			// G = Gconstant(iter+1, iteraciones);
			// G-=0.24f;
			G-=G_step;

			cout<<"Gconstant iter "<< iter+1<<" = " <<Gconstant(iter+1, iteraciones)<<endl;

		//(e)Calculation of the total force in different directions.
			//***********  sort by masses
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
			for (auto &agent : poblacion)  //agent == particle
			{
				for (size_t i = 0; i < dimension; i++)
				{
					if (agent.Mi == 0)
						agent.acc[i] = 0;
					else
						agent.acc[i] = agent.forces[i] / agent.Mi;     //Equation 10
				}
			}

			for (auto &agent : poblacion)
			{
				float randi = distribucion(generador);
				for (size_t i = 0; i < dimension; i++)
					agent.velocity[i] = randi * agent.velocity[i] + agent.acc[i];  //Equation 11
			}

		//(g)Updating agents’ position.

		for (auto &agent : poblacion)
		{
			for (size_t i = 0; i < dimension; i++)
			{
				agent.position[i] = std::min<float>(agent.position[i] + agent.velocity[i], this->max);
				agent.position[i] = std::max<float>(agent.position[i], this->min);  //Equation 12
			}
		}

		//(h)Repeat steps c to g until the stop criteria is reached.
	}

	cout<<"bestPositionSoFar: ";

	for (int i = 0; i < dimension; ++i)
		std::cout<<bestSoFarPosition[i] <<" ";

	cout<<"bestFitnessSoFar: "<<bestSoFarFitness<<endl;

	//(i)End.
	return 1;
}

//Equation 7
std::vector<float> Gravitational::F(Agente i, Agente j)
{
	float R = 0.0f;
	int dim = i.position.size();
	for (size_t component = 0; component < dim; component++)
		R += pow(i.position[component] - j.position[component],2);

	R = sqrt(R);

	float e = 2.2204e-016;//algun valor pequeño
	float sub_F = i.Mp*j.Ma / (R + e);

	std::vector<float> Force;
	for (size_t component = 0; component < dim; component++)
		Force.push_back(sub_F*(j.position[component] - i.position[component]));

	return Force;
}
