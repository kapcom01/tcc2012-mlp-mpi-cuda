#ifndef HOSTEXAMPLESET_H_
#define HOSTEXAMPLESET_H_

#include "arff/Relation.h"
#include "mlp/Types.h"

#define BIG_M 1000000

namespace ParallelMLP
{

/**
 * Classe que contém um conjunto de dados experimentais na CPU
 */
class HostExampleSet
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
	void normalize();

	/**
	 * Desnormaliza as entradas, saídas alvo e saídas do conjunto de dados
	 */
	void unnormalize();

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
	void setOutput(uint i, float* output);

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
	 * Índice de inserção em input
	 */
	uint inputIdx;

	/**
	 * Dados de saída da rede neural
	 */
	float* output;

	/**
	 * Índice de inserção em output
	 */
	uint outputIdx;

	/**
	 * Estatísticas para cada coluna de dados
	 */
	Stat* stat;

	/**
	 * Índice de inserção em stat
	 */
	uint statIdx;

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
