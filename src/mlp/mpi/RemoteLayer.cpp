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
	counts.resize(hosts);
	offset.resize(hosts);

	// Realiza cálculos para determinar o balanceamento
	RemoteUtils::balance(outUnits, counts, offset);

	HostLayer::init(inUnits, counts[hid]);
}

//===========================================================================//

RemoteLayer::~RemoteLayer()
{

}

//===========================================================================//

}

