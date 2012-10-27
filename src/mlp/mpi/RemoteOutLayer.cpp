#include "mlp/mpi/RemoteOutLayer.h"

namespace ParallelMLP
{

//===========================================================================//

RemoteOutLayer::RemoteOutLayer(uint inUnits, uint outUnits, uint hid,
		uint hosts)
	: Layer(inUnits, outUnits), OutLayer(inUnits, outUnits)
{
	this->hid = hid;

	// Aloca os vetores
	weights = new float[connUnits];
	gradient = new float[outUnits];
	funcSignal = new float[outUnits + 1];
	errorSignal = new float[inUnits];

	funcSignal[outUnits] = 1;
}

//===========================================================================//

RemoteOutLayer::~RemoteOutLayer()
{
	delete[] weights;
	delete[] gradient;
	delete[] funcSignal;
	delete[] errorSignal;
}

//===========================================================================//

void RemoteOutLayer::randomize()
{
	if (hid == 0)
		for (uint i = 0; i < connUnits; i++)
			weights[i] = random();
}

//===========================================================================//

void RemoteOutLayer::feedforward(const float* input)
{
	// Apenas o mestre executa o feedforward
	if (hid == 0)
	{
		this->input = input;

		// Inicializa o sinal funcional
		memset(funcSignal, 0, outUnits * sizeof(float));

		// Calcula o sinal funcional
		for (uint i = 0; i < connUnits; i++)
		{
			uint j = i % inUnits;
			uint k = i / inUnits;
			funcSignal[k] += weights[i] * input[j];
		}

		// Ativa o sinal funcional
		for (uint i = 0; i < outUnits; i++)
			funcSignal[i] = activate(funcSignal[i]);
	}
}

//===========================================================================//

void RemoteOutLayer::calculateError(const float* target)
{
	float sum = 0;

	// Calcula o erro cometido pela rede
	for (uint i = 0; i < outUnits; i++)
	{
		error[i] = target[i] - funcSignal[i];
		sum += error[i] * error[i];
	}

	// Incrementa o erro
	incError(sum);
}

//===========================================================================//

void RemoteOutLayer::feedbackward(const float* target, float learning)
{
	// Apenas o mestre executa o feedbackward
	if (hid == 0)
	{
		calculateError(target);

		// Inicializa o sinal funcional
		memset(errorSignal, 0, (inUnits - 1) * sizeof(float));

		// Calcula o gradiente
		for (uint i = 0; i < outUnits; i++)
			gradient[i] = derivate(funcSignal[i]) * target[i];

		// Atualiza os pesos e calcula o sinal de erro
		for (uint i = 0; i < connUnits; i++)
		{
			uint j = i % inUnits;
			uint k = i / inUnits;
			weights[i] += learning * gradient[k] * input[j];
			errorSignal[j] += gradient[k] * weights[i];
		}
	}

	// Envia o sinal de erro do mestre para os escravos
	COMM_WORLD.Bcast(&totalError, 1, FLOAT, 0);
	COMM_WORLD.Bcast(errorSignal, inUnits - 1, FLOAT, 0);
}

//===========================================================================//

float RemoteOutLayer::random() const
{
	float r = rand() / (float) RAND_MAX;
	return 2 * r - 1;
}

//===========================================================================//

float RemoteOutLayer::activate(float x) const
{
	return tanh(x);
}

//===========================================================================//

float RemoteOutLayer::derivate(float y) const
{
	return (1 - y) * (1 + y);
}

//===========================================================================//

}

