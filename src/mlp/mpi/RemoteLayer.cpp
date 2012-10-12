#include "mlp/mpi/RemoteLayer.h"

namespace ParallelMLP
{

//===========================================================================//

RemoteLayer::RemoteLayer()
{
	hid = -1;
}

//===========================================================================//

RemoteLayer::RemoteLayer(uint inUnits, uint outUnits, uint hid, uint hosts)
{
	init(inUnits, outUnits, hid, hosts);
}

//===========================================================================//

void RemoteLayer::init(uint inUnits, uint outUnits, uint hid, uint hosts)
{
	this->hid = hid;

	// Realiza cálculos para determinar o balanceamento
	binfo.resize(hosts);
	binfo.balance(outUnits);

	HostLayer::init(inUnits, outUnits);

	// Aumenta o tamanho do sinal funcional
	funcSignal.resize(outUnits);
	rawFuncSignal = vec_float(funcSignal);
}

//===========================================================================//

RemoteLayer::~RemoteLayer()
{

}

//===========================================================================//

void RemoteLayer::feedforward(const vec_float &input)
{
	uint offset = binfo.getOffset(hid);

	// Seta o offset para realizar os cálculos corretamente
	rawFuncSignal.setOffset(offset);

	HostLayer::feedforward(input);

	// Retira o offset
	rawFuncSignal.setOffset(-offset);

	// Recebe os dados de todos os sinais funcionais
	COMM_WORLD.Allgatherv(rawFuncSignal(offset), binfo.getCount(hid), FLOAT,
			rawFuncSignal(0), binfo.getCounts(), binfo.getOffsets(), FLOAT);
}

//===========================================================================//

void RemoteLayer::feedback(const vec_float &signal, float learning)
{
	uint offset = binfo.getOffset(hid);
	vec_float csignal = signal;

	// Seta o offset para realizar os cálculos corretamente
	csignal.setOffset(offset);

	HostLayer::feedback(csignal, learning);

	// Retira o offset
	csignal.setOffset(-offset);

	// Recebe os dados de todos os sinais de erro
	COMM_WORLD.Allreduce(rawErrorSignal(0), rawErrorSignal(0), inUnits, FLOAT,
			SUM);
}

//===========================================================================//

}

