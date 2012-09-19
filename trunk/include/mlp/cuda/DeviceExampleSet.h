#ifndef DEVICEEXAMPLESET_H_
#define DEVICEEXAMPLESET_H_

#include "mlp/common/ExampleSet.h"
#include "mlp/Vector.h"
#include <thrust/sort.h>

namespace ParallelMLP
{

/**
 * Classe que contém um conjunto de dados experimentais no dispositivo
 */
class DeviceExampleSet : public ExampleSet
{

public:

	/**
	 * Constrói um conjunto de dados vazio
	 * @param relationID ID da relação
	 * @param mlpID ID da rede
	 * @param type Tipo do conjunto de dados
	 */
	DeviceExampleSet(int relationID, int mlpID, SetType type);

	/**
	 * Destrói o conjunto de dados
	 */
	virtual ~DeviceExampleSet();

	/**
	 * Copia os dados da memória da CPU para a memória da GPU
	 */
	void copyToDevice();

	/**
	 * Copia os dados da memória da GPU para a memória da CPU
	 */
	void copyToHost();

	/**
	 * Normaliza as entradas e saídas alvo do conjunto de dados
	 */
	void normalize();

	/**
	 * Desnormaliza as entradas, saídas alvo e saídas do conjunto de dados
	 */
	void unnormalize();

private:

	/**
	 * Dados de entrada do treinamento
	 */
	dv_float devInput;

	/**
	 * Ponteiro para os dados de entrada e seu tamanho
	 */
	Vector<float> rawInput;

	/**
	 * Dados de saída alvo para o treinamento
	 */
	dv_float devTarget;

	/**
	 * Ponteiro para os dados de saída alvo e seu tamanho
	 */
	Vector<float> rawTarget;

	/**
	 * Dados de saída da rede neural
	 */
	dv_float devOutput;

	/**
	 * Ponteiro para os dados de saída da rede e seu tamanho
	 */
	Vector<float> rawOutput;

	/**
	 * Estatísticas para cada coluna de entrada
	 */
	dv_stat devInStat;

	/**
	 * Ponteiro para as estatísticas de entrada e seu tamanho
	 */
	Vector<Stat> rawInStat;

	/**
	 * Estatísticas para cada coluna de saída
	 */
	dv_stat devOutStat;

	/**
	 * Ponteiro para os estatísticas de saída e seu tamanho
	 */
	Vector<Stat> rawOutStat;

};

}

#endif
