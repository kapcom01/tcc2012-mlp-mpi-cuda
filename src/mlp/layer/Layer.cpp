#include "mlp/layer/Layer.h"

namespace MLP
{

//===========================================================================//

Layer::Layer(uint inUnits, uint outUnits, const ActivationFunc *activation)
{
	this->activation = activation;
	this->inUnits = inUnits;
	this->outUnits = outUnits;

	// Aloca a matriz de pesos
	weights = new double*[outUnits];
	for (uint i = 0; i < outUnits; i++)
		weights[i] = new double[inUnits + 1];

	// Aloca os vetores de saída, de feedback e de erro
	weightedSum = new double[outUnits];
	output = new double[outUnits];
	feedbackSignal = new double[inUnits];
	error = new double[outUnits];
}

//===========================================================================//

Layer::~Layer()
{
	// Deleta a matriz de pesos
	for (uint i = 0; i < outUnits; i++)
		delete[] weights[i];
	delete[] weights;

	// Deleta os vetores de saída, de feedback e de erro
	delete[] weightedSum;
	delete[] output;
	delete[] feedbackSignal;
	delete[] error;
}

//===========================================================================//

void Layer::randomizeWeights()
{
	// Inicializa os pesos das entradas
	for (uint i = 0; i < outUnits; i++)
		for (uint j = 0; j <= inUnits; j++)
			weights[i][j] = activation->initialValue(inUnits, outUnits);
}

//===========================================================================//

void Layer::feedforward(const double* input)
{
	this->input = input;

	// Para cada neurônio
	for (uint i = 0; i < outUnits; i++)
	{
		double* row = weights[i];
		double sum = 0;

		// Para cada entrada
		for (uint j = 0; j < inUnits; j++)
			sum += row[j] * input[j];
		sum += row[inUnits];

		// Seta as saídas não ativada e ativada
		weightedSum[i] = sum;
		output[i] = activation->activate(sum);
	}
}

//===========================================================================//

void Layer::feedback(const double* signal, double learningRate)
{
	// Atualiza os pesos dos neurônios
	for (uint i = 0; i < outUnits; i++)
	{
		// Calcula o erro do neurônio atual
		error[i] = calculateError(i, signal);

		double* row = weights[i];

		// Atualiza os pesos desse neurônio
		for (uint j = 0; j < inUnits; j++)
			row[j] += learningRate * error[i] * input[j];
		row[inUnits] += learningRate * error[i];
	}

	// Calcula o sinal de retorno para a camada anterior
	for (uint j = 0; j < inUnits; j++)
	{
		double sum = 0;

		// Calcula a soma dos erros ponderados pelos pesos de entrada
		for (uint i = 0; i < outUnits; i++)
			sum += error[i] * weights[i][j];

		feedbackSignal[j] = sum;
	}
}

//===========================================================================//

}

