#ifndef EXAMPLESETADAPTER_H_
#define EXAMPLESETADAPTER_H_

#include "mlp/common/ExampleSet.h"
#include "database/RelationAdapter.h"
#include <algorithm>

namespace ParallelMLP
{

/**
 * Estrutura que representa o tamanho de uma relação
 */
struct Size
{
	/**
	 * Quantidade de instâncias
	 */
	uint nInst;

	/**
	 * Quantidade de atributos
	 */
	uint nAttr;
};

/**
 * Classe responsável por realizar operações sobre um conjunto de dados na base
 */
class ExampleSetAdapter
{

public:

	/**
	 * Recupera um conjunto de dados
	 * @param set Conjunto de dados a ser selecionado
	 */
	static void select(ExampleSet &set);

	/**
	 * Insere um conjunto de dados
	 * @param set Conjunto de dados
	 */
	static void insert(const ExampleSet &set);

private:

	/**
	 * Prepara a conexão para operações de seleção
	 * @param conn Conexão
	 */
	static void prepareForSelect(connection* conn);

	/**
	 * Prepara a conexão para operações de seleção
	 * @param conn Conexão
	 */
	static void prepareForInsert(connection* conn);

	/**
	 * Seleciona a quantidade de instâncias e atributos de uma relação
	 * @param relationID Identificador da relação
	 * @param work Trabalho
	 * @return Quantidade de instâncias e atributos de uma relação
	 */
	static Size selectSize(int relationID, WorkPtr &work);

	/**
	 * Seleciona os dados de uma relação
	 * @param set Conjunto de dados a serem preenchidos
	 * @param size Quantidade de instâncias e atributos da relação
	 * @param work Trabalho
	 */
	static void selectData(ExampleSet &set, Size &size, WorkPtr &work);

	/**
	 * Seleciona a relação de treinamento
	 * @param set Conjunto de dados
	 * @return ID da relação de treinamento
	 */
	static int selectTrainedRelation(ExampleSet &set, WorkPtr &work);

	/**
	 * Seleciona o intervalo de valores do MLP
	 * @param mlpID ID da rede
	 * @param work Trabalho
	 * @return Intervalo de valores do MLP
	 */
	static Range selectRange(int mlpID, WorkPtr &work);

	/**
	 * Seleciona as estatísticas de uma relação
	 * @param set Conjunto de dados a serem preenchidos
	 * @param size Quantidade de instâncias e atributos da relação
	 * @param work Trabalho
	 */
	static void selectStatistics(ExampleSet &set, Size &size, WorkPtr &work);

	/**
	 * Insere os dados da operação
	 * @param set Conjunto de dados
	 * @param work Trabalho
	 * @return ID da operação
	 */
	static int insertOperation(const ExampleSet &set, WorkPtr &work);

	/**
	 * Verifica o tipo do atributo de saída
	 * @param set Conjunto de dados
	 * @param work Trabalho
	 * @return Retorna verdadeiro se for numérico e falso caso contrário
	 */
	static bool selectType(const ExampleSet &set, WorkPtr &work);

	/**
	 * Insere os resultados de uma operação
	 * @param opID ID da operação
	 * @param set Conjunto de dados
	 * @param work Trabalho
	 */
	static void insertResults(int opID, const ExampleSet &set, WorkPtr &work);

};

}

#endif
