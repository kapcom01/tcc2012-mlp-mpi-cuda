#include "mlp/serial/HostNeuron.h"

namespace ParallelMLP
{

float random();

float activate(float x);

float derivate(float y);

//===========================================================================//

HostNeuron::HostNeuron(uint inUnits, float &cOutput, hv_float &cError)
	: Neuron(inUnits), output(cOutput), error(cError)
{

}

//===========================================================================//

HostNeuron::~HostNeuron()
{

}

//===========================================================================//

void HostNeuron::randomize()
{
	for (uint i = 0; i <= inUnits; i++)
		weights[i] = random();
}

//===========================================================================//

void HostNeuron::execute(const vec_float input)
{
	for (uint i = 0; i < inUnits; i++)
		output += input[i] * weights[i];
	output += weights[inUnits];

	output = activate(output);
}

//===========================================================================//

void HostNeuron::response(const vec_float input, float signal, float learning)
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

float random()
{
	float r = rand() / (float) RAND_MAX;
	return 2 * r - 1;
}

//===========================================================================//

float activate(float x)
{
	return tanh(x);
}

//===========================================================================//

float derivate(float y)
{
	return (1 - y) * (1 + y);
}

//===========================================================================//

}
