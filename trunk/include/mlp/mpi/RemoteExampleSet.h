#ifndef REMOTEEXAMPLESET_H_
#define REMOTEEXAMPLESET_H_

#include "mlp/serial/HostExampleSet.h"
#include "mlp/mpi/BalanceInfo.h"
#include <mpi.h>

using namespace MPI;

namespace ParallelMLP
{

/**
 * Classe que contém um conjunto de dados experimentais
 */
class RemoteExampleSet : public HostExampleSet
{

public:

	/**
	 * Constrói um conjunto de dados vazio
	 * @param relationID ID da relação
	 * @param mlpID ID da rede
	 * @param type Tipo do conjunto de dados
	 * @param hid ID do host
	 * @param hosts Quantidade de hosts
	 */
	RemoteExampleSet(int relationID, int mlpID, SetType type, uint hid,
			uint hosts);

	/**
	 * Destrói o conjunto de dados
	 */
	virtual ~RemoteExampleSet();

	/**
	 * Normaliza as entradas e saídas alvo do conjunto de dados
	 */
	virtual void normalize();

	/**
	 * Desnormaliza as entradas, saídas alvo e saídas do conjunto de dados
	 */
	virtual void unnormalize();

	/**
	 * Finaliza a inserção de dados
	 */
	virtual void done();

protected:

	/**
	 * Altera o tamanho dos vetores
	 */
	void resize();

	/**
	 * ID do host
	 */
	uint hid;

	/**
	 * Informações do balanceamento
	 */
	BalanceInfo binfo;

};

}

#endif
