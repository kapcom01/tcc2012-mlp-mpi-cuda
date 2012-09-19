#ifndef HOSTEXAMPLESET_H_
#define HOSTEXAMPLESET_H_

#include "mlp/common/ExampleSet.h"

namespace ParallelMLP
{

/**
 * Classe que contém um conjunto de dados experimentais
 */
class HostExampleSet : public ExampleSet
{

public:

	/**
	 * Constrói um conjunto de dados vazio
	 * @param relationID ID da relação
	 * @param mlpID ID da rede
	 * @param type Tipo do conjunto de dados
	 */
	HostExampleSet(int relationID, int mlpID, SetType type);

	/**
	 * Destrói o conjunto de dados
	 */
	virtual ~HostExampleSet();

	/**
	 * Normaliza as entradas e saídas alvo do conjunto de dados
	 */
	void normalize();

	/**
	 * Desnormaliza as entradas, saídas alvo e saídas do conjunto de dados
	 */
	void unnormalize();

};

}

#endif
