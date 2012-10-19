#include "mlp/common/Indexes.h"

namespace ParallelMLP
{

//===========================================================================//

Indexes::Indexes()
{

}

//===========================================================================//

void Indexes::resize(uint size)
{
	indexes.resize(size);

	// Inicia o vetor de índices
	for (uint i = 0; i < size; i++)
		indexes[i] = i;
}

//===========================================================================//

void Indexes::randomize()
{
	for (uint i = indexes.size() - 1; i > 0; i--)
	{
		// Troca o valor da posição i com o de uma posição aleatória
		uint j = rand() % (i + 1);
		swap(indexes[i], indexes[j]);
	}
}

//===========================================================================//

uint Indexes::get(uint i) const
{
	return indexes[i];
}

//===========================================================================//

}
