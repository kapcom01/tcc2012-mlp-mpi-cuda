#include "mlp/mpi/RemoteUtils.h"

namespace ParallelMLP
{

//===========================================================================//

bool RemoteUtils::isMaster(uint hid)
{
	return (hid == MASTER);
}

//===========================================================================//

bool RemoteUtils::isSlave(uint hid)
{
	return (hid != MASTER);
}

//===========================================================================//

void RemoteUtils::balance(uint total, v_int &counts, v_int &offset)
{
	// Cria um vetor contendo a quantidade de unidades por cada host
	for (uint i = 0; i < counts.size(); i++)
		counts[i] = total / counts.size();

	// Incrementa o restante nos hosts iniciais para melhor balanceamento
	for (uint i = 0; i < total % counts.size(); i++)
		counts[i]++;

	// Calcula qual será a unidade inicial para este host
	offset[0] = 0;
	for (uint i = 1; i < counts.size(); i++)
		offset[i] = offset[i - 1] + counts[i - 1];
}

//===========================================================================//

}

