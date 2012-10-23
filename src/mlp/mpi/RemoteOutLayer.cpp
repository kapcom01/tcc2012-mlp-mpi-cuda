#include "mlp/mpi/RemoteOutLayer.h"

namespace ParallelMLP
{

//===========================================================================//

RemoteOutLayer::RemoteOutLayer(uint inUnits, uint outUnits, uint hid,
		uint hosts)
	: Layer(inUnits, outUnits), HostOutLayer(inUnits, outUnits)
{
	this->hid = hid;
}

//===========================================================================//

void RemoteOutLayer::feedforward(const float* input)
{
	// Apenas o mestre executa o feedforward
	if (hid == 0)
		HostOutLayer::feedforward(input);
}

//===========================================================================//

void RemoteOutLayer::feedbackward(const float* signal, float learning)
{
	// Apenas o mestre executa o feedbackward
	if (hid == 0)
		HostOutLayer::feedbackward(signal, learning);

	// Envia o sinal de erro do mestre para os escravos
	COMM_WORLD.Bcast(errorSignal, inUnits - 1, FLOAT, 0);
}

//===========================================================================//

}

