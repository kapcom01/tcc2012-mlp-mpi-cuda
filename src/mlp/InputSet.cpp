#include "mlp/InputSet.h"

namespace MLP
{

//===========================================================================//

InputSet::InputSet(uint size, uint inVars, uint outVars)
{
	this->size = size;
	this->inVars = inVars;
	this->outVars = outVars;

	// Constrói as linhas dos valores
	input = new double*[size];
	expectedOutput = new double*[size];
	output = new double*[size];

	// Constrói as colunas dos valores
	for (uint i = 0; i < size; i++)
	{
		input[i] = new double[inVars];
		expectedOutput[i] = new double[outVars];
		output[i] = new double[outVars];
	}
}

//===========================================================================//

InputSet::~InputSet()
{
	// Destrói as colunas dos valores
	for (uint i = 0; i < size; i++)
	{
		delete[] input[i];
		delete[] expectedOutput[i];
		delete[] output[i];
	}

	// Destrói as linhas dos valores
	delete[] input;
	delete[] expectedOutput;
	delete[] output;
}

}
