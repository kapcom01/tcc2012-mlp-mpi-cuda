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
	if (hid == 0)
	{
		weights = new float[connUnits];
		bias = new float[outUnits];
		gradient = new float[outUnits];
		funcSignal = new float[outUnits];
		error = new float[outUnits];
	}

	errorSignal = new float[inUnits];
}

//===========================================================================//

RemoteOutLayer::~RemoteOutLayer()
{
	if (hid == 0)
	{
		delete[] weights;
		delete[] bias;
		delete[] gradient;
		delete[] funcSignal;
		delete[] error;
	}

	delete[] errorSignal;
}

//===========================================================================//

void RemoteOutLayer::randomize()
{
	if (hid == 0)
	{
		for (uint i = 0; i < connUnits; i++)
			weights[i] = random();

		for (uint i = 0; i < outUnits; i++)
			bias[i] = random();
	}
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
			funcSignal[i] = ACTIVATE(bias[i] + funcSignal[i]);
	}
}

//===========================================================================//

void RemoteOutLayer::calculateError(const float* target)
{
	if (hid == 0)
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
}

//===========================================================================//

void RemoteOutLayer::feedbackward(const float* target, float learning)
{
	// Apenas o mestre executa o feedbackward
	if (hid == 0)
	{
		calculateError(target);

		// Inicializa o sinal funcional
		memset(errorSignal, 0, inUnits * sizeof(float));

		// Calcula o gradiente
		for (uint i = 0; i < outUnits; i++)
		{
			gradient[i] = DERIVATE(funcSignal[i]) * target[i];
			bias[i] += gradient[i];
		}

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
	COMM_WORLD.Bcast(errorSignal, inUnits, FLOAT, 0);
}

//===========================================================================//

float RemoteOutLayer::random() const
{
	float r = rand() / (float) RAND_MAX;
	return 2 * r - 1;
}

//===========================================================================//

}

