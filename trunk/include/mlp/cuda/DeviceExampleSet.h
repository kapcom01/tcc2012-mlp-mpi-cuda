#ifndef DEVICEEXAMPLESET_H_
#define DEVICEEXAMPLESET_H_

#include "mlp/common/ExampleSet.h"
#include "mlp/Vector.h"
#include <thrust/sort.h>

#define MAX_BLOCKS 256

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

	/**
	 * Retorna a i-ésima entrada do conjunto
	 * @param i Índice da entrada
	 * @return Entrada de índice i
	 */
	vec_float getInput(uint i);

	/**
	 * Retorna a i-ésima saída alvo do conjunto
	 * @param i Índice da saída alvo
	 * @return Saída alvo de índice i
	 */
	vec_float getTarget(uint i);

	/**
	 * Seta os valores da i-ésima saída
	 * @param output Vetor contendo a i-ésima saída
	 */
	void setOutput(uint i, const vec_float output);

private:

	/**
	 * Dados de entrada do treinamento
	 */
	dv_float devInput;

	/**
	 * Ponteiro para os dados de entrada e seu tamanho
	 */
	vec_float rawInput;

	/**
	 * Dados de saída da rede neural
	 */
	dv_float devOutput;

	/**
	 * Ponteiro para os dados de saída da rede e seu tamanho
	 */
	vec_float rawOutput;

	/**
	 * Estatísticas para cada coluna de dados
	 */
	dv_stat devStat;

	/**
	 * Ponteiro para as estatísticas e seu tamanho
	 */
	vec_stat rawStat;

};

}

#endif
