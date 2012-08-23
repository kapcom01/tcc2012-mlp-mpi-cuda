#ifndef EXAMPLESET_H_
#define EXAMPLESET_H_

#include "Common.h"

namespace Database { class ExampleSetAdapter; }

namespace MLP
{

/**
 * Classe que contém os dados de entrada
 */
class ExampleSet
{

public:

	/**
	 * Constrói um conjunto de entrada vazio
	 */
	ExampleSet();

	/**
	 * Destrói o conjunto de entradas
	 */
	virtual ~ExampleSet();

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
	const vector<double>& getInput(uint i) const;

	/**
	 * Retorna a i-ésima saída alvo do conjunto
	 * @param i Índice da saída alvo
	 * @return Saída alvo de índice i
	 */
	const vector<double>& getTarget(uint i) const;

	friend class Database::ExampleSetAdapter;
	friend class BackpropMLP;

//private:

	/**
	 * Adiciona uma nova instância
	 */
	void pushInstance();

	/**
	 * Adiciona um valor númerico de entrada ou saída
	 * @param value Valor numérico de entrada ou saída
	 * @param isTarget Indica se o valor é de saída
	 */
	void addValue(const double &value, bool isTarget);

	/**
	 * Adiciona um valor nominal de entrada ou saída
	 * @param value Valor nominal de entrada ou saída
	 * @param card Cardinalidade do atributo nominal
	 * @param isTarget Indica se o valor é de saída
	 */
	void addValue(const int &value, const uint &card, bool isTarget);

	/**
	 * Taxa de aprendizado
	 */
	double learningRate;

	/**
	 * Momento
	 */
	double momentum;

	/**
	 * Número máximo de épocas
	 */
	uint maxEpochs;

	/**
	 * Tolerância máxima
	 */
	double maxTolerance;

	/**
	 * Taxa de sucesso mínima
	 */
	double minSuccessRate;

	/**
	 * Dados de entrada do treinamento
	 */
	vector<vector<double>> input;

	/**
	 * Dados de saída alvo para o treinamento
	 */
	vector<vector<double>> target;

	/**
	 * Dados de saída da rede neural
	 */
	vector<vector<double>> output;

	/**
	 * Taxa de sucesso
	 */
	double successRate;

};

/**
 * Ponteiro para InputSet
 */
typedef shared_ptr<ExampleSet> InputSetPtr;

}

#endif
