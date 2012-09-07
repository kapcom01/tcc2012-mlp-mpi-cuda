#ifndef BACKPROPMLPADAPTER_H_
#define BACKPROPMLPADAPTER_H_

#include "mlp/BackpropMLP.h"
#include "database/Connection.h"

using namespace MLP;

namespace Database
{

/**
 * Classe responsável por realizar operações sobre um MLP na base de dados
 */
class BackpropMLPAdapter
{

public:

	/**
	 * Insere um MLP
	 * @param mlp Multi-Layer Perceptron
	 */
	static void insert(BackpropMLP &mlp);

	/**
	 * Recupera um MLP
	 * @param mlp Multi-Layer Perceptron
	 */
	static void select(BackpropMLP &mlp);

	/**
	 * Atualiza um MLP
	 * @param mlp Multi-Layer Perceptron
	 * @param relationID Relação com qual a rede foi treinada
	 */
	static void update(const BackpropMLP &mlp, int relationID);

private:

	/**
	 * Prepara a conexão para operações de inserção
	 * @param conn Conexão
	 */
	static void prepareForInsert(connection* conn);

	/**
	 * Prepara a conexão para operações de seleção
	 * @param conn Conexão
	 */
	static void prepareForSelect(connection* conn);

	/**
	 * Prepara a conexão para operações de atualização
	 * @param conn Conexão
	 */
	static void prepareForUpdate(connection* conn);

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
	 * @param work Trabalho
	 * @return ID gerado
	 */
	static int insertMLP(const BackpropMLP &mlp, WorkPtr &work);

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
	 * Insere as informações de um neurônio
	 * @param mlpID ID da rede
	 * @param layerIndex Índice da camada
	 * @param neuronIndex Índice do neurônio
	 * @param neuron Neurônio
	 * @param work Trabalho
	 */
	static void insertNeuron(int mlpID, uint layerIndex, uint neuronIndex,
			const Neuron &neuron, WorkPtr &work);

	/**
	 * Seleciona as informações do MLP
	 * @param mlp Multi-Layer Perceptron
	 * @param work Trabalho
	 */
	static void selectMLP(BackpropMLP &mlp, WorkPtr &work);

	/**
	 * Seleciona os pesos de uma camada
	 * @param mlpID ID da rede
	 * @param layerIndex Índice da camada
	 * @param layer Camada
	 * @param work Trabalho
	 */
	static void selectLayer(int mlpID, uint layerIndex, Layer &layer,
			WorkPtr &work);

	/**
	 * Seleciona as informações de um neurônio
	 * @param mlpID ID da rede
	 * @param layerIndex Índice da camada
	 * @param neuronIndex Índice do neurônio
	 * @param neuron Neurônio
	 * @param work Trabalho
	 */
	static void selectNeuron(int mlpID, uint layerIndex, uint neuronIndex,
			Neuron &neuron, WorkPtr &work);

	/**
	 * Adiciona a relação com qual a rede foi treinada
	 * @param mlpID ID da rede
	 * @param relationID Relação com qual a rede foi treinada
	 * @param work Trabalho
	 */
	static void updateRelation(int mlpID, int relationID, WorkPtr &work);

	/**
	 * Atualiza os pesos de uma camada
	 * @param mlpID ID da rede
	 * @param layerIndex Índice da camada
	 * @param layer Camada
	 * @param work Trabalho
	 */
	static void updateLayer(int mlpID, uint layerIndex, const Layer &layer,
			WorkPtr &work);

	/**
	 * Atualiza as informações de um neurônio
	 * @param mlpID ID da rede
	 * @param layerIndex Índice da camada
	 * @param neuronIndex Índice do neurônio
	 * @param neuron Neurônio
	 * @param work Trabalho
	 */
	static void updateNeuron(int mlpID, uint layerIndex, uint neuronIndex,
			const Neuron &neuron, WorkPtr &work);

};

}

#endif
