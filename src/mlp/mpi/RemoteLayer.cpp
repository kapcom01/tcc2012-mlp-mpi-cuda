#include "mlp/mpi/RemoteLayer.h"

namespace ParallelMLP
{

//===========================================================================//

RemoteLayer::RemoteLayer(uint inUnits, uint outUnits, uint hid, uint hosts)
	: Layer(inUnits, outUnits)
{
	this->hid = hid;

	// Realiza cálculos para determinar o balanceamento
	binfo.resize(hosts);
	binfo.balance(outUnits);
	toutUnits = binfo.getCount(hid);
	tconnUnits = (inUnits + 1) * toutUnits;
	offset = binfo.getOffset(hid);

	// Aloca os vetores
	weights = new float[tconnUnits];
	gradient = new float[toutUnits];
	tfuncSignal = new float[toutUnits];
	funcSignal = new float[outUnits + 1];
	errorSignal = new float[inUnits];

	funcSignal[outUnits] = 1;
}

//===========================================================================//

RemoteLayer::~RemoteLayer()
{
	delete[] weights;
	delete[] gradient;
	delete[] funcSignal;
	delete[] tfuncSignal;
	delete[] errorSignal;
}

//===========================================================================//

void RemoteLayer::randomize()
{
	for (uint i = 0; i < tconnUnits; i++)
		weights[i] = random();
}

//===========================================================================//

void RemoteLayer::feedforward(const float* input)
{
	this->input = input;

	// Inicializa o sinal funcional
	memset(tfuncSignal, 0, toutUnits * sizeof(float));

	// Calcula o sinal funcional
	for (uint i = 0; i < tconnUnits; i++)
	{
		uint j = i % inUnits;
		uint k = i / inUnits;
		tfuncSignal[k] += weights[i] * input[j];
	}

	// Ativa o sinal funcional
	for (uint i = 0; i < toutUnits; i++)
		tfuncSignal[i] = ACTIVATE(tfuncSignal[i]);

	// Recebe os dados de todos os sinais funcionais
	COMM_WORLD.Allgatherv(tfuncSignal, toutUnits, FLOAT, funcSignal,
			binfo.getCounts(), binfo.getOffsets(), FLOAT);
}

//===========================================================================//

void RemoteLayer::feedbackward(const float* signal, float learning)
{
	// Inicializa o sinal funcional
	memset(errorSignal, 0, (inUnits - 1) * sizeof(float));

	// Calcula o gradiente
	for (uint i = 0; i < toutUnits; i++)
		gradient[i] = DERIVATE(funcSignal[i]) * signal[i + offset];

	// Atualiza os pesos e calcula o sinal de erro
	for (uint i = 0; i < tconnUnits; i++)
	{
		uint j = i % inUnits;
		uint k = i / inUnits;
		weights[i] += learning * gradient[k] * input[j];
		errorSignal[j] += gradient[k] * weights[i];
	}

	// Recebe os dados de todos os sinais de erro
	COMM_WORLD.Allreduce(errorSignal, errorSignal, inUnits - 1, FLOAT, SUM);
}

//===========================================================================//

float RemoteLayer::random() const
{
	float r = rand() / (float) RAND_MAX;
	return 2 * r - 1;
}

//===========================================================================//

}

