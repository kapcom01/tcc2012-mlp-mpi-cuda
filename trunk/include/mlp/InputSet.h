#ifndef INPUTSET_H_
#define INPUTSET_H_

#include "Common.h"

namespace Database { class RelationAdapter; }

namespace MLP
{

/**
 * Classe que contém os dados de treinamento
 */
class InputSet
{

public:

	/**
	 * Constrói um conjunto de entrada vazio
	 */
	InputSet();

	/**
	 * Destrói o conjunto de entradas
	 */
	virtual ~InputSet();

	/**
	 * Retorna a quantidade de variáveis de entrada
	 * @return Quantidade de variáveis de entrada
	 */
	uint inVars();

	/**
	 * Retorna a quantidade de variáveis de saída
	 * @return Quantidade de variáveis de saída
	 */
	uint outVars();

	/**
	 * Retorna o tamanho do conjunto de entrada
	 * @return Tamanho do conjunto de entrada
	 */
	uint size();

	friend class Database::RelationAdapter;

private:

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
typedef shared_ptr<InputSet> InputSetPtr;

}

#endif
