#include "mlp/mpi/RemoteExampleSet.h"

namespace ParallelMLP
{

//===========================================================================//

RemoteExampleSet::RemoteExampleSet(int relationID, int mlpID, SetType type,
		uint hid, uint hosts)
	: HostExampleSet(relationID, mlpID, type)
{
	this->hid = hid;
	binfo.resize(hosts);
}

//===========================================================================//

RemoteExampleSet::~RemoteExampleSet()
{

}

//===========================================================================//

void RemoteExampleSet::resize()
{
	input.resize(size * (inVars + outVars));

	if (hid == 0)
		output.resize(size * outVars);
	else
		output.resize(size * binfo.getCount(hid));
}

//===========================================================================//

void RemoteExampleSet::normalize()
{
	// Normalização é feita apenas no mestre
	if (hid == 0)
		HostExampleSet::normalize();

	// Transmite a quantidade de variáveis e instâncias
	COMM_WORLD.Bcast(&inVars, 1, INT, 0);
	COMM_WORLD.Bcast(&outVars, 1, INT, 0);
	COMM_WORLD.Bcast(&size, 1, INT, 0);

	// Altera o tamanho dos vetores nos escravos
	if (hid != 0)
		resize();

	// Transmite os dados
	COMM_WORLD.Bcast(&input[0], input.size(), FLOAT, 0);
}

//===========================================================================//

void RemoteExampleSet::unnormalize()
{
	// Desnormalização é feita apenas no mestre
	if (hid == 0)
		HostExampleSet::unnormalize();
}

//===========================================================================//

void RemoteExampleSet::done()
{
	HostExampleSet::done();
	binfo.balance(outVars);
}

//===========================================================================//

}
