#ifndef EXAMPLESET_H_
#define EXAMPLESET_H_

#include "mlp/Vector.h"

namespace ParallelMLP
{

/**
 * Tipos de conjunto de dados
 */
enum SetType
{
	TRAINING = 1, VALIDATION = 2, TEST = 3
};

/**
 * Classe que contém um conjunto de dados experimentais
 */
class ExampleSet
{

public:

	/**
	 * Constrói um conjunto de dados vazio
	 * @param relationID ID da relação
	 * @param mlpID ID da rede
	 * @param type Tipo do conjunto de dados
	 */
	ExampleSet(int relationID, int mlpID, SetType type);

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
	 * Seta algumas propriedades do conjunto de amostras
	 * @param learning Taxa de aprendizado
	 * @param maxEpochs Quantidade máxima de épocas
	 * @param tolerance Tolerância
	 */
	void setProperties(float learning, uint maxEpochs, float tolerance);

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
	 * Retorna a i-ésima entrada do conjunto
	 * @param i Índice da entrada
	 * @return Entrada de índice i
	 */
	const hv_float getInput(uint i) const;

	/**
	 * Retorna a i-ésima saída alvo do conjunto
	 * @param i Índice da saída alvo
	 * @return Saída alvo de índice i
	 */
	const hv_float getTarget(uint i) const;

	/**
	 * Seta os valores da i-ésima saída
	 * @param output Vetor contendo a i-ésima saída
	 */
	void setOutput(uint i, const hv_float &output);

	friend class ExampleSetAdapter;
	friend class BackpropMLP;

protected:

	void print();

	/**
	 * ID da relação
	 */
	int relationID;

	/**
	 * ID da rede
	 */
	int mlpID;

	/**
	 * Tipo do conjunto de dados
	 */
	SetType type;

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
	 * Dados de entrada do treinamento
	 */
	hv_float input;

	/**
	 * Dados de saída alvo para o treinamento
	 */
	hv_float target;

	/**
	 * Dados de saída da rede neural
	 */
	hv_float output;

	/**
	 * Estatísticas para cada coluna de entrada
	 */
	hv_stat inStat;

	/**
	 * Estatísticas para cada coluna de saída
	 */
	hv_stat outStat;

	/**
	 * Erro cometido pela rede
	 */
	float error;

	/**
	 * Tempo de execução da operação
	 */
	float time;

	/**
	 * Indica se o conjunto de dados está normalizado
	 */
	bool isNormalized;

};

}

#endif
