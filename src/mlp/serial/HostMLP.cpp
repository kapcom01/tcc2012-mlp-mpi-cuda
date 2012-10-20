#include "mlp/serial/HostMLP.h"

namespace ParallelMLP
{

//===========================================================================//

HostMLP::HostMLP(v_uint &units)
	: MLP(units)
{
	// Adiciona as camadas escondidas e a camada de saída
	for (uint i = 0; i < units.size() - 1; i++)
	{
		if (i + 1 == units.size() - 1)
			layers.push_back(new HostOutLayer(units[i], units[i + 1]));
		else
			layers.push_back(new HostLayer(units[i], units[i + 1]));
	}

	// Seta os ponteiros para a primeira e última camada
	linkLayers();
}

//===========================================================================//

HostMLP::~HostMLP()
{

}

//===========================================================================//

}
