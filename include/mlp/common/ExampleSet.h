#ifndef EXAMPLESET_H_
#define EXAMPLESET_H_

#include "arff/Relation.h"
#include "mlp/Types.h"

#define BIG_M 1000000

namespace ParallelMLP
{

/**
 * Classe que contém um conjunto de dados experimentais
 */
class ExampleSet
{

public:

	/**
	 * Constrói um conjunto de dados a partir de uma relação
	 * @param relation Relação
	 */
	ExampleSet(const Relation& relation);

	/**
	 * Destrói o conjunto de dados
	 */
	virtual ~ExampleSet();

	/**
	 * Normaliza as entradas e saídas alvo do conjunto de dados
	 */
	virtual void normalize() = 0;

	/**
	 * Desnormaliza as entradas, saídas alvo e saídas do conjunto de dados
	 */
	virtual void unnormalize() = 0;

	/**
	 * Retorna a quantidade de variáveis de entrada
	 * @return Quantidade de variáveis de entrada
	 */
	uint getInVars() const;

	/**
	 * Retorna a quantidade de variáveis de saída
	 * @return Quantidade de variáveis de saída
	 */
	uint getOutVars() const;

	/**
	 * Retorna o tamanho do conjunto de entrada
	 * @return Tamanho do conjunto de entrada
	 */
	uint getSize() const;

	/**
	 * Retorna todas as entradas
	 * @return Todas as entradas
	 */
	const float* getInput() const;

	/**
	 * Retorna a i-ésima entrada do conjunto
	 * @param i Índice da entrada
	 * @return Entrada de índice i
	 */
	const float* getInput(uint i) const;

	/**
	 * Retorna a i-ésima saída alvo do conjunto
	 * @param i Índice da saída alvo
	 * @return Saída alvo de índice i
	 */
	const float* getTarget(uint i) const;

	/**
	 * Seta os valores da i-ésima saída
	 * @param output Vetor contendo a i-ésima saída
	 */
	virtual void setOutput(uint i, float* output) = 0;

	/**
	 * Retorna todas as estatísticas
	 * @return Todas as estatísticas
	 */
	const Stat* getStat() const;

	/**
	 * Retorna a taxa de aprendizado
	 * @return Taxa de aprendizado
	 */
	float getLearning() const;

	/**
	 * Seta a taxa de aprendizado
	 * @param learning Taxa de aprendizado
	 */
	void setLearning(float learning);

	/**
	 * Retorna a tolerância
	 * @return Tolerância
	 */
	float getTolerance() const;

	/**
	 * Seta a tolerância
	 * @param tolerance Tolerância
	 */
	void setTolerance(float tolerance);

	/**
	 * Retorna a quantidade máxima de épocas
	 * @return Quantidade máxima de épocas
	 */
	uint getMaxEpochs() const;

	/**
	 * Seta a quantidade máxima de épocas
	 * @param maxEpochs Quantidade máxima de épocas
	 */
	void setMaxEpochs(uint maxEpochs);

	/**
	 * Retorna o erro quadrático médio cometido
	 * @return Erro quadrático médio cometido
	 */
	float getError() const;

	/**
	 * Seta o erro quadrático médio cometido
	 * @param error Erro quadrático médio cometido
	 */
	void setError(float error);

	/**
	 * Retorna a quantidade de épocas utilizadas
	 * @return Quantidade de épocas utilizadas
	 */
	uint getEpochs() const;

	/**
	 * Seta a quantidade de épocas utilizadas
	 * @param epochs Quantidade de épocas utilizadas
	 */
	void setEpochs(uint epochs);

	/**
	 * Retorna o tempo da operação em milisegundos
	 * @return Tempo da operação em milisegundos
	 */
	float getTime() const;

	/**
	 * Seta o tempo da operação em milisegundos
	 * @param time Tempo da operação em milisegundos
	 */
	void setTime(float time);

protected:

	/**
	 * Taxa de aprendizado
	 */
	float learning;

	/**
	 * Número máximo de épocas
	 */
	uint maxEpochs;

	/**
	 * Tolerância máxima
	 */
	float tolerance;

	/**
	 * Quantidade de variáveis de entrada
	 */
	uint inVars;

	/**
	 * Quantidade de variáveis de saída
	 */
	uint outVars;

	/**
	 * Quantidade de instâncias
	 */
	uint size;

	/**
	 * Quantidade de variáveis por instância
	 */
	uint step;

	/**
	 * Dados de entrada do treinamento
	 */
	float* input;

	/**
	 * Dados de saída da rede neural
	 */
	float* output;

	/**
	 * Estatísticas para cada coluna de dados
	 */
	Stat* stat;

	/**
	 * Erro cometido pela rede
	 */
	float error;

	/**
	 * Tempo de execução da operação
	 */
	float time;

	/**
	 * Quantidade de iterações feitas
	 */
	uint epochs;

	/**
	 * Indica se o conjunto de dados está normalizado
	 */
	bool isNormalized;

};

}

#endif
