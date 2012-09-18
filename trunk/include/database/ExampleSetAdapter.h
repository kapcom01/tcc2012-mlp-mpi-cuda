#ifndef EXAMPLESETADAPTER_H_
#define EXAMPLESETADAPTER_H_

#include "mlp/ExampleSet.h"
#include "database/RelationAdapter.h"
#include <algorithm>

namespace ParallelMLP
{

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
	 * Seleciona a quantidade de atributos de uma relação
	 * @param relationID Identificador da relação
	 * @param work Trabalho
	 * @return Quantidade de atributos de uma relação
	 */
	static int selectNAttributes(int relationID, WorkPtr &work);

	/**
	 * Seleciona os dados de uma relação
	 * @param set Conjunto de dados a serem preenchidos
	 * @param nattr Quantidade de atributos da relação
	 * @param work Trabalho
	 */
	static void selectData(ExampleSet &set, int nattr, WorkPtr &work);

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
	 * @param nattr Quantidade de atributos da relação
	 * @param work Trabalho
	 */
	static void selectStatistics(ExampleSet &set, WorkPtr &work);

	/**
	 * Adiciona um valor númerico de entrada ou saída
	 * @param set Conjunto de dados
	 * @param value Valor numérico de entrada ou saída
	 * @param isTarget Indica se o valor é de saída
	 */
	static void addValue(ExampleSet &set, double value, bool isTarget);

	/**
	 * Adiciona um valor nominal de entrada ou saída
	 * @param set Conjunto de dados
	 * @param value Valor nominal de entrada ou saída
	 * @param card Cardinalidade do atributo nominal
	 * @param isTarget Indica se o valor é de saída
	 */
	static void addValue(ExampleSet &set, int value, uint card, bool isTarget);

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
