#include "mlp/serial/Neuron.h"

namespace ParallelMLP
{

//===========================================================================//

Neuron::Neuron(uint inUnits, float &cOutput, hv_float &cError)
	: output(cOutput), error(cError)
{
	this->inUnits = inUnits;

	weights.resize(inUnits + 1);
	gradient = 0;
}

//===========================================================================//

Neuron::~Neuron()
{

}

//===========================================================================//

void Neuron::randomize()
{
	for (uint i = 0; i <= inUnits; i++)
		weights[i] = random();
}

//===========================================================================//

void Neuron::execute(const hv_float &input)
{
	for (uint i = 0; i < inUnits; i++)
		output += input[i] * weights[i];
	output += weights[inUnits];

	output = activate(output);
}

//===========================================================================//

void Neuron::response(const hv_float &input, float signal, float learning)
{
	gradient = derivate(output) * signal;

	for (uint i = 0; i < inUnits; i++)
	{
		weights[i] += learning * gradient * input[i];
		error[i] += gradient * weights[i];
	}
	weights[inUnits] += learning * gradient;
}

//===========================================================================//

float Neuron::random() const
{
	float r = rand() / (float) RAND_MAX;
	return 2 * r - 1;
}

//===========================================================================//

float Neuron::activate(float x) const
{
	return tanh(x);
}

//===========================================================================//

float Neuron::derivate(float y) const
{
	return (1 - y) * (1 + y);
}

//===========================================================================//

}
