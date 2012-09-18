#include "mlp/Layer.h"

namespace ParallelMLP
{

//===========================================================================//

Layer::Layer(uint inUnits, uint outUnits)
{
	this->inUnits = inUnits;
	this->outUnits = outUnits;
	this->input = NULL;

	// Aloca os vetores de saída, de feedback e de erro
	funcSignal.resize(outUnits);
	errorSignal.resize(inUnits);

	// Adiciona os neurônios
	for (uint n = 0; n < outUnits; n++)
	{
		NeuronPtr neuron(new Neuron(inUnits, funcSignal[n], errorSignal));
		neurons.push_back(neuron);
	}
}

//===========================================================================//

Layer::~Layer()
{

}

//===========================================================================//

void Layer::randomize()
{
	for (uint n = 0; n < outUnits; n++)
		neurons[n]->randomize();
}

//===========================================================================//

void Layer::feedforward(const vdouble& input)
{
	this->input = &(input);

	// Inicializa o sinal funcional
	thrust::fill(funcSignal.begin(), funcSignal.end(), 0);

	// Executa a ativação de cada neurônio
	for (uint n = 0; n < outUnits; n++)
		neurons[n]->execute(input);
}

//===========================================================================//

void Layer::feedback(const vdouble &signal, double learning)
{
	// Inicializa o sinal funcional
	thrust::fill(errorSignal.begin(), errorSignal.end(), 0);

	// Atualiza os pesos de cada neurônio
	for (uint n = 0; n < outUnits; n++)
		neurons[n]->response(*input, signal[n], learning);
}

//===========================================================================//

}

