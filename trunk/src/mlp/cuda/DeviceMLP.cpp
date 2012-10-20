#include "mlp/cuda/DeviceMLP.h"

namespace ParallelMLP
{

//===========================================================================//

DeviceMLP::DeviceMLP(v_uint &units)
	: MLP(units)
{
	// Adiciona as camadas escondidas e a camada de saída
	for (uint i = 0; i < units.size() - 1; i++)
	{
		if (i + 1 == units.size() - 1)
			layers.push_back(new DeviceOutLayer(units[i], units[i + 1]));
		else
			layers.push_back(new DeviceLayer(units[i], units[i + 1]));
	}

	// Seta os ponteiros para a primeira e última camada
	linkLayers();
}

//===========================================================================//

DeviceMLP::~DeviceMLP()
{

}

//===========================================================================//


}
