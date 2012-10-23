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
class RemoteExampleSet : public ExampleSet
{

public:

	/**
	 * Constrói um conjunto de dados a partir de uma relação
	 * @param relation Relação
	 * @param hid ID do host
	 */
	RemoteExampleSet(const Relation &relation, uint hid);

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
	 * Seta os valores da i-ésima saída
	 * @param output Vetor contendo a i-ésima saída
	 */
	virtual void setOutput(uint i, float* output);

protected:

	/**
	 * Copia os dados experimentais para o mestre
	 */
	void copyToMaster(const HostExampleSet &set);

	/**
	 * Ajusta um valor de um range para outro
	 * @param x Valor a ser ajustado
	 * @param from Range inicial de x
	 * @param to Range final de x
	 */
	void adjust(float &x, const Range &from, const Range &to) const;

	/**
	 * ID do host
	 */
	uint hid;

};

}

#endif
