#include "mlp/serial/HostLayer.h"

namespace ParallelMLP
{

//===========================================================================//

HostLayer::HostLayer(uint inUnits, uint outUnits)
	: Layer(inUnits, outUnits)
{
	// Adiciona os neurônios
	for (uint n = 0; n < outUnits; n++)
		neurons.push_back(new HostNeuron(inUnits, funcSignal[n], errorSignal));
}

//===========================================================================//

HostLayer::~HostLayer()
{

}

//===========================================================================//

void HostLayer::feedforward(const vec_float input)
{
	this->input = input;

	// Inicializa o sinal funcional
	thrust::fill(funcSignal.begin(), funcSignal.end(), 0);

	// Executa a ativação de cada neurônio
	for (uint n = 0; n < outUnits; n++)
		neurons[n]->execute(input);
}

//===========================================================================//

void HostLayer::feedback(const vec_float signal, float learning)
{
	// Inicializa o sinal funcional
	thrust::fill(errorSignal.begin(), errorSignal.end(), 0);

	// Atualiza os pesos de cada neurônio
	for (uint n = 0; n < outUnits; n++)
		neurons[n]->response(input, signal[n], learning);
}

//===========================================================================//

}

