#include "mlp/mpi/BalanceInfo.h"

namespace ParallelMLP
{

//===========================================================================//

BalanceInfo::BalanceInfo()
{

}

//===========================================================================//

BalanceInfo::~BalanceInfo()
{

}

//===========================================================================//

void BalanceInfo::resize(uint hosts)
{
	counts.resize(hosts);
	offsets.resize(hosts);
}

//===========================================================================//

void BalanceInfo::balance(uint total)
{
	// Cria um vetor contendo a quantidade de unidades por cada host
	for (uint i = 0; i < counts.size(); i++)
		counts[i] = total / counts.size();

	// Incrementa o restante nos hosts iniciais para melhor balanceamento
	for (uint i = 0; i < total % counts.size(); i++)
		counts[i]++;

	// Calcula qual será a unidade inicial para este host
	offsets[0] = 0;
	for (uint i = 1; i < counts.size(); i++)
		offsets[i] = offsets[i - 1] + counts[i - 1];
}

//===========================================================================//

int* BalanceInfo::getCounts()
{
	return &counts[0];
}

//===========================================================================//

int BalanceInfo::getCount(uint hid)
{
	return counts[hid];
}

//===========================================================================//

int* BalanceInfo::getOffsets()
{
	return &offsets[0];
}

//===========================================================================//

int BalanceInfo::getOffset(uint hid)
{
	return offsets[hid];
}

//===========================================================================//

}

