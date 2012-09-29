#ifndef MLPADAPTER_H_
#define MLPADAPTER_H_

#include "mlp/common/MLP.h"
#include "database/Connection.h"

namespace ParallelMLP
{

/**
 * Classe responsável por realizar operações sobre um MLP na base de dados
 */
class MLPAdapter
{

public:

	/**
	 * Insere um MLP
	 * @param mlp Multi-Layer Perceptron
	 */
	static void insert(MLP &mlp);

	/**
	 * Recupera um MLP
	 * @param mlp Multi-Layer Perceptron
	 */
	static void select(MLP &mlp);

	/**
	 * Atualiza um MLP
	 * @param mlp Multi-Layer Perceptron
	 * @param relationID Relação com qual a rede foi treinada
	 */
	static void update(const MLP &mlp, int relationID);

private:

	/**
	 * Prepara a conexão para operações de inserção
	 * @param conn Conexão
	 */
	static void prepareForInsert(connection &conn);

	/**
	 * Verifica se o nome do MLP já existe anteriormente
	 * @param name Nome do MLP
	 * @param work Trabalho
	 * @return Verdadeiro se não existir ou falso caso contrário
	 */
	static bool checkUnique(const string &name, work &work);

	/**
	 * Insere as informações básicas do MLP
	 * @param mlp Multi-Layer Perceptron
	 * @param work Trabalho
	 * @return ID gerado
	 */
	static int insertMLP(const MLP &mlp, work &work);

	/**
	 * Insere as informações de uma camada
	 * @param mlpID ID da rede
	 * @param layerIndex Índice da camada
	 * @param layer Camada
	 * @param work Trabalho
	 */
	static void insertLayer(int mlpID, uint layerIndex, const Layer &layer,
			work &work);

	/**
	 * Prepara a conexão para operações de seleção
	 * @param conn Conexão
	 */
	static void prepareForSelect(connection &conn);

	/**
	 * Seleciona as informações do MLP
	 * @param mlp Multi-Layer Perceptron
	 * @param work Trabalho
	 */
	static void selectMLP(MLP &mlp, work &work);

	/**
	 * Seleciona os pesos de uma camada
	 * @param mlpID ID da rede
	 * @param layerIndex Índice da camada
	 * @param layer Camada
	 * @param work Trabalho
	 */
	static void selectLayer(int mlpID, uint layerIndex, Layer &layer,
			work &work);

	/**
	 * Prepara a conexão para operações de atualização
	 * @param conn Conexão
	 */
	static void prepareForUpdate(connection &conn);

	/**
	 * Adiciona a relação com qual a rede foi treinada
	 * @param mlpID ID da rede
	 * @param relationID Relação com qual a rede foi treinada
	 * @param work Trabalho
	 */
	static void updateRelation(int mlpID, int relationID, work &work);

	/**
	 * Atualiza os pesos de uma camada
	 * @param mlpID ID da rede
	 * @param layerIndex Índice da camada
	 * @param layer Camada
	 * @param work Trabalho
	 */
	static void updateLayer(int mlpID, uint layerIndex, const Layer &layer,
			work &work);

};

}

#endif
