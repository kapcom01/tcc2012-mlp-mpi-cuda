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
	HostOutLayer::init(inUnits, outUnits);
	this->hid = hid;
}

//===========================================================================//

RemoteOutLayer::~RemoteOutLayer()
{

}

//===========================================================================//

void RemoteOutLayer::feedforward(const vec_float &input)
{
	if (RemoteUtils::isMaster(hid))
		HostOutLayer::feedforward(input);
}

//===========================================================================//

void RemoteOutLayer::feedback(const vec_float &signal, float learning)
{
	if (RemoteUtils::isMaster(hid))
		HostOutLayer::feedback(signal, learning);
}

//===========================================================================//

}

