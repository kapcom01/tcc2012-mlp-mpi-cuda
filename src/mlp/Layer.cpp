#include "mlp/Layer.h"

namespace MLP
{

//===========================================================================//

Layer::Layer(uint inUnits, uint outUnits, const ActivationFunc &cActivation)
	: activation(cActivation)
{
	this->inUnits = inUnits;
	this->outUnits = outUnits;
	input = NULL;

	// Aloca a matriz de pesos
	weights.resize(outUnits);
	delta.resize(outUnits);
	for (uint i = 0; i < outUnits; i++)
	{
		weights[i].resize(inUnits + 1);
		delta[i].resize(inUnits + 1);
	}

	// Aloca os vetores de saída, de feedback e de erro
	funcSignal.resize(outUnits);
	errorSignal.resize(inUnits);
	gradient.resize(outUnits);
}

//===========================================================================//

Layer::~Layer()
{

}

//===========================================================================//

void Layer::randomizeWeights()
{
	// Inicializa os pesos das entradas
	for (uint i = 0; i < outUnits; i++)
		for (uint j = 0; j <= inUnits; j++)
		{
			weights[i][j] = randomWeight();
			delta[i][j] = 0;
		}
}

//===========================================================================//

void Layer::feedforward(const vector<double>& input)
{
	this->input = &input;

	// Para cada neurônio
	for (uint i = 0; i < outUnits; i++)
	{
		double sum = 0;

		// Para cada entrada
		for (uint j = 0; j < inUnits; j++)
			sum += weights[i][j] * input[j];
		sum += weights[i][inUnits];

		// Seta as saídas não ativada e ativada
		funcSignal[i] = activation.activate(sum);
	}
}

//===========================================================================//

void Layer::feedback(const vector<double> &signal, double learningRate,
		double momentum)
{
	// Atualiza os pesos dos neurônios
	for (uint i = 0; i < outUnits; i++)
	{
		// Calcula o gradiente local
		gradient[i] = activation.derivate(funcSignal[i]) * signal[i];

		// Atualiza os pesos desse neurônio utilizando o momento
		for (uint j = 0; j < inUnits; j++)
		{
			delta[i][j] = momentum * delta[i][j]
					+ learningRate * gradient[i] * input->at(j);
			weights[i][j] += delta[i][j];
//			weights[i][j] += learningRate * gradient[i] * input->at(j);
		}
		delta[i][inUnits] = momentum * delta[i][inUnits]
				+ learningRate * gradient[i];
		weights[i][inUnits] += delta[i][inUnits];
//		weights[i][inUnits] += learningRate * gradient[i];
	}

	// Calcula o sinal de retorno para a camada anterior
	for (uint j = 0; j < inUnits; j++)
	{
		double sum = 0;

		// Calcula a soma dos erros ponderados pelos pesos de entrada
		for (uint i = 0; i < outUnits; i++)
			sum += gradient[i] * weights[i][j];

		errorSignal[j] = sum;
	}
}

//===========================================================================//

double Layer::randomWeight() const
{
	double r = rand() / (double) RAND_MAX;
	return MAX_INIT_WEIGHT * (2 * r - 1);
}

//===========================================================================//

}

