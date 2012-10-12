#include "mlp/mpi/RemoteOutLayer.h"

namespace ParallelMLP
{

//===========================================================================//

RemoteOutLayer::RemoteOutLayer()
{
	this->hid = -1;
}

//===========================================================================//

RemoteOutLayer::RemoteOutLayer(uint inUnits, uint outUnits, uint hid,
		uint hosts)
{
	init(inUnits, outUnits, hid, hosts);
}

//===========================================================================//

void RemoteOutLayer::init(uint inUnits, uint outUnits, uint hid, uint hosts)
{
	this->hid = hid;

	// Inicializa a camada se for o nó mestre
	if (hid == 0)
		HostOutLayer::init(inUnits, outUnits);
	// Caso for algum nó escravo, aloca somente espaço para o sinal de erro
	else
	{
		errorSignal.resize(inUnits);
		rawErrorSignal = vec_float(errorSignal);
	}
}

//===========================================================================//

RemoteOutLayer::~RemoteOutLayer()
{

}

//===========================================================================//

void RemoteOutLayer::feedforward(const vec_float &input)
{
	if (hid == 0)
		HostOutLayer::feedforward(input);
}

//===========================================================================//

void RemoteOutLayer::feedback(const vec_float &signal, float learning)
{
	if (hid == 0)
		HostOutLayer::feedback(signal, learning);

	// Envia o sinal de erro do mestre para os escravos
	COMM_WORLD.Bcast(rawErrorSignal(0), inUnits, FLOAT, 0);
}

//===========================================================================//

}

