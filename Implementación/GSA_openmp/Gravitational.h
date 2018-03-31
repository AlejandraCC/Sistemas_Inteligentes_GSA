#pragma once
struct Agente
{
	std::vector<float> data,forces,vel,acc;
	float Ma, Mp, Mi,m;
	float fitness;
	Agente(int dim,std::random_device &generador,std::uniform_real_distribution<float> &distribucion)
	{
		for (size_t i = 0; i < dim; i++)
		{
			data.push_back(distribucion(generador));
			forces.push_back(0.0f);
			vel.push_back(0.0f);
			acc.push_back(0.0f);
		}
		m=Ma = Mp = Mi = fitness = -1.0f;
	}
};

class Gravitational
{
private:
	std::vector<Agente> poblacion;
	unsigned int dimension;
	float min, max;
	bool minimizacion;
	float(*funcion)(std::vector<float>);
	
public:
	float best_data[50];
	Gravitational(float(*funcion)(std::vector<float>), unsigned int numero_poblacion, unsigned int dimension, float minimo, float maximo, bool minimizar);
	~Gravitational();
	// optimizacion
	float optimizar(int iteraciones);
	std::vector<float> F(Agente i, Agente j);
};

