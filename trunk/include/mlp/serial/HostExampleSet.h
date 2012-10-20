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
	 * Adiciona um valor estatístico numérico de entrada ou saída
	 * @param min Valor mínimo da amostra
	 * @param max Valor máximo da amostra
	 * @param lower Menor valor depois de normalizado
	 * @param upper Maior valor depois de normalizado
	 * @param isTarget Indica se o valor é de saída
	 */
	void addStat(float min, float max, float lower, float upper,
			bool isTarget);

	/**
	 * Adiciona um valor estatístico nominal de entrada ou saída
	 * @param lower Menor valor depois de normalizado
	 * @param upper Maior valor depois de normalizado
	 * @param card Cardinalidade do atributo nominal
	 * @param isTarget Indica se o valor é de saída
	 */
	void addStat(float lower, float upper, uint card, bool isTarget);

	/**
	 * Seta os dados a partir de uma relação
	 * @param relation Relação
	 */
	void setRelation(const Relation& relation);

	/**
	 * Ajusta um valor de um range para outro
	 * @param x Valor a ser ajustado
	 * @param from Range inicial de x
	 * @param to Range final de x
	 */
	void adjust(float &x, const Range &from, const Range &to) const;

};

}

#endif
