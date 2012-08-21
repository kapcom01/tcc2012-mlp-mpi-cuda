#ifndef MLPHELPER_H_
#define MLPHELPER_H_

#include "mlp/MLP_BP.h"
#include "database/Connection.h"

using namespace MLP;

namespace Database
{

/**
 * Classe responsável por realizar operações sobre um MLP na base de dados
 */
class MLPHelper
{

public:

	/**
	 * Insere um MLP
	 * @param mlp Multi-Layer Perceptron
	 * @param name Nome da rede
	 */
	static void insert(BackpropMLP &mlp, const string &name);

	/**
	 * Atualiza um MLP
	 * @param mlp Multi-Layer Perceptron
	 */
	static void update(const BackpropMLP &mlp);

private:

	/**
	 * Prepara a conexão para algumas operações
	 * @param conn Conexão
	 */
	static void prepare(connection* conn);

	/**
	 * Verifica se o nome do MLP já existe anteriormente
	 * @param name Nome do MLP
	 * @param work Trabalho
	 * @return Verdadeiro se não existir ou falso caso contrário
	 */
	static bool checkUnique(const string &name, WorkPtr &work);

	/**
	 * Insere as informações básicas do MLP
	 * @param mlp Multi-Layer Perceptron
	 * @param name Nome da rede
	 * @param work Trabalho
	 * @return ID gerado
	 */
	static int insertMLP(const BackpropMLP &mlp, const string &name,
			WorkPtr &work);

	/**
	 * Insere as informações de uma camada
	 * @param mlpID ID da rede
	 * @param layerIndex Índice da camada
	 * @param layer Camada
	 * @param work Trabalho
	 */
	static void insertLayer(int mlpID, uint layerIndex, const Layer &layer,
			WorkPtr &work);

	/**
	 * Insere os pesos de um neurônio
	 * @param mlpID ID da rede
	 * @param layerIndex Índice da camada
	 * @param neuronIndex Índice do neurônio
	 * @param weights Vetor contendo os pesos do neurônio
	 * @param inUnits Quantidade de variáveis de entrada
	 * @param work Trabalho
	 */
	static void insertNeuron(int mlpID, uint layerIndex, uint neuronIndex,
			const double* weights, uint inUnits, WorkPtr &work);

	/**
	 * Atualiza os pesos de um neurônio
	 * @param mlpID ID da rede
	 * @param layerIndex Índice da camada
	 * @param neuronIndex Índice do neurônio
	 * @param weights Vetor contendo os pesos do neurônio
	 * @param inUnits Quantidade de variáveis de entrada
	 * @param work Trabalho
	 */
	static void updateNeuron(int mlpID, uint layerIndex, uint neuronIndex,
			const double* weights, uint inUnits, WorkPtr &work);

};

}

#endif
