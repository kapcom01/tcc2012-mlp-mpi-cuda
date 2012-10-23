#include "mlp/mpi/RemoteMLP.h"

namespace ParallelMLP
{

//===========================================================================//

RemoteMLP::RemoteMLP(v_uint &units)
	: MLP(units)
{
	hid = COMM_WORLD.Get_rank();
	hosts = COMM_WORLD.Get_size();

	// Adiciona as camadas escondidas e a camada de saída
	for (uint i = 0; i < units.size() - 1; i++)
	{
		if (i + 1 == units.size() - 1)
			layers.push_back(new RemoteOutLayer(units[i], units[i + 1],
					hid, hosts));
		else
			layers.push_back(new RemoteLayer(units[i], units[i + 1],
					hid, hosts));
	}

	// Seta os ponteiros para a primeira e última camada
	linkLayers();
}

//===========================================================================//

RemoteMLP::~RemoteMLP()
{

}

//===========================================================================//

}
