#include "mlp/mpi/RemoteExampleSet.h"

namespace ParallelMLP
{

//===========================================================================//

RemoteExampleSet::RemoteExampleSet(int relationID, int mlpID, SetType type,
		uint hid, uint hosts)
	: HostExampleSet(relationID, mlpID, type)
{
	this->hid = hid;
	counts.resize(hosts);
	offset.resize(hosts);
}

//===========================================================================//

RemoteExampleSet::~RemoteExampleSet()
{

}

//===========================================================================//

void RemoteExampleSet::resize()
{
	input.resize(size * (inVars + outVars));

	if (RemoteUtils::isMaster(hid))
		output.resize(size * outVars);
	else
		output.resize(size * counts[hid]);
}

//===========================================================================//

void RemoteExampleSet::normalize()
{
	// Normalização é feita apenas no mestre
	if (RemoteUtils::isMaster(hid))
		HostExampleSet::normalize();

	// Transmite a quantidade de variáveis e instâncias
	COMM_WORLD.Bcast(&inVars, 1, INT, RemoteUtils::MASTER);
	COMM_WORLD.Bcast(&outVars, 1, INT, RemoteUtils::MASTER);
	COMM_WORLD.Bcast(&size, 1, INT, RemoteUtils::MASTER);

	// Altera o tamanho dos vetores nos escravos
	if (RemoteUtils::isSlave(hid))
		resize();

	// Transmite os dados
	COMM_WORLD.Bcast(&input[0], input.size(), FLOAT, RemoteUtils::MASTER);
}

//===========================================================================//

void RemoteExampleSet::unnormalize()
{
	// Desnormalização é feita apenas no mestre
	if (RemoteUtils::isMaster(hid))
		HostExampleSet::unnormalize();
}

//===========================================================================//

void RemoteExampleSet::done()
{
	HostExampleSet::done();
	RemoteUtils::balance(outVars, counts, offset);
}

//===========================================================================//

}
