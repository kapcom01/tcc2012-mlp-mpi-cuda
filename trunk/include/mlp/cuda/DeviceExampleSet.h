#ifndef DEVICEEXAMPLESET_H_
#define DEVICEEXAMPLESET_H_

#include "mlp/serial/HostExampleSet.h"

namespace ParallelMLP
{

/**
 * Classe que contém um conjunto de dados experimentais na GPU
 */
class DeviceExampleSet : public ExampleSet
{

public:

	/**
	 * Constrói um conjunto de dados a partir de uma relação
	 * @param relation Relação
	 */
	DeviceExampleSet(const Relation& relation);

	/**
	 * Destrói o conjunto de dados
	 */
	virtual ~DeviceExampleSet();

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
	 * Copia os dados da memória da CPU para a memória da GPU
	 * @param set Conjunto de dados na CPU
	 */
	void copyToDevice(const HostExampleSet &set);

	/**
	 * Quantidade de blocos para execução de um kernel em função do passo
	 */
	uint stepBlocks;

	/**
	 * Quantidade de blocos para execução de um kernel em função do saída
	 */
	uint outBlocks;

};

}

#endif
