#ifndef EXAMPLESET_H_
#define EXAMPLESET_H_

#include "mlp/Types.h"

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
 * Estrutura que armazena um intervalo
 */
struct Range
{
	/**
	 * Valor mínimo
	 */
	double lower;

	/**
	 * Valor máximo
	 */
	double upper;
};

/**
 * Estrutura contendo estatísticas sobre o conjunto de entrada
 */
struct Statistics
{
	/**
	 * Intervalo de valores de um dado
	 */
	Range from;

	/**
	 * Intervalo de valores de um dado normalizado
	 */
	Range to;
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
	 * @param range Intervalo de valores para a nova saída alvo
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
	uint inVars() const;

	/**
	 * Retorna a quantidade de variáveis de saída
	 * @return Quantidade de variáveis de saída
	 */
	uint outVars() const;

	/**
	 * Retorna o tamanho do conjunto de entrada
	 * @return Tamanho do conjunto de entrada
	 */
	uint size() const;

	/**
	 * Retorna a i-ésima entrada do conjunto
	 * @param i Índice da entrada
	 * @return Entrada de índice i
	 */
	const vdouble& getInput(uint i) const;

	/**
	 * Retorna a i-ésima saída alvo do conjunto
	 * @param i Índice da saída alvo
	 * @return Saída alvo de índice i
	 */
	const vdouble& getTarget(uint i) const;

	friend class ExampleSetAdapter;
	friend class BackpropMLP;

//private:

	/**
	 * Ajusta o valor de x para um novo intervalo
	 * @param x Valor a ser ajustado
	 * @param from Intervalo de valores para x
	 * @param to Intervalo de valores para a saída
	 */
	void adjust(double &x, const Range &from, const Range &to);

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
	double learning;

	/**
	 * Número máximo de épocas
	 */
	uint maxEpochs;

	/**
	 * Tolerância máxima
	 */
	double tolerance;

	/**
	 * Dados de entrada do treinamento
	 */
	vector<vdouble> input;

	/**
	 * Dados de saída alvo para o treinamento
	 */
	vector<vdouble> target;

	/**
	 * Dados de saída da rede neural
	 */
	vector<vdouble> output;

	/**
	 * Estatísticas para cada coluna de dados
	 */
	vector<Statistics> stat;

	/**
	 * Erro cometido pela rede
	 */
	double error;

	/**
	 * Tempo de execução da operação
	 */
	double time;

	/**
	 * Indica se o conjunto de dados está normalizado
	 */
	bool isNormalized;

};

/**
 * Ponteiro para InputSet
 */
typedef shared_ptr<ExampleSet> InputSetPtr;

}

#endif
