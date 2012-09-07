#include "mlp/Neuron.h"

namespace MLP
{

//===========================================================================//

Neuron::Neuron(uint inUnits, double &cOutput, vdouble &cError)
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

void Neuron::execute(const vdouble &input)
{
	for (uint i = 0; i < inUnits; i++)
		output += input[i] * weights[i];
	output += weights[inUnits];

	output = activate(output);
}

//===========================================================================//

void Neuron::response(const vdouble &input, double signal, double learning)
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

double Neuron::random() const
{
	double r = rand() / (double) RAND_MAX;
	return 2 * r - 1;
}

//===========================================================================//

double Neuron::activate(double x) const
{
	return tanh(x);
}

//===========================================================================//

double Neuron::derivate(double y) const
{
	return (1 - y) * (1 + y);
}

//===========================================================================//

}
