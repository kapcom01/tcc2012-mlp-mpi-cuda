#ifndef HOSTEXAMPLESET_H_
#define HOSTEXAMPLESET_H_

#include "mlp/common/ExampleSet.h"

namespace ParallelMLP
{

/**
 * Classe que contém um conjunto de dados experimentais na CPU
 */
class HostExampleSet : public ExampleSet
{

public:

	/**
	 * Constrói um conjunto de dados com dados aleatórios
	 * @param size Quantidade de instâncias
	 * @param inVars Quantidade de variáveis de entrada
	 * @param outVars Quantidade de variáveis de entrada
	 */
	HostExampleSet(uint size, uint inVars, uint outVars);

	/**
	 * Constrói um conjunto de dados a partir de uma relação
	 * @param relation Relação
	 */
	HostExampleSet(const Relation& relation);

	/**
	 * Destrói o conjunto de dados
	 */
	virtual ~HostExampleSet();

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
	 * Inicializa o conjunto de dados
	 */
	void init();

	/**
	 * Randomiza os dados
	 */
	void randomize();

	/**
	 * Seta os dados a partir de uma relação
	 * @param relation Relação
	 */
	void setRelation(const Relation& relation);

	/**
	 * Adiciona as estatísticas dos dados
	 */
	void addStatistics();

	/**
	 * Adiciona um valor bias
	 */
	void addBias();

	/**
	 * Adiciona um valor númerico de entrada ou saída
	 * @param value Valor numérico de entrada ou saída
	 * @param isTarget Indica se o valor é de saída
	 */
	void addValue(float value, bool isTarget);

	/**
	 * Adiciona um valor nominal de entrada ou saída
	 * @param value Valor nominal de entrada ou saída
	 * @param card Cardinalidade do atributo nominal
	 * @param isTarget Indica se o valor é de saída
	 */
	void addValue(int value, uint card, bool isTarget);

	/**
	 * Ajusta um valor de um range para outro
	 * @param x Valor a ser ajustado
	 * @param from Range inicial de x
	 * @param to Range final de x
	 */
	void adjust(float &x, const Range &from, const Range &to) const;

	/**
	 * Índice de inserção em input
	 */
	uint inputIdx;

	/**
	 * Índice de inserção em output
	 */
	uint outputIdx;

};

}

#endif
