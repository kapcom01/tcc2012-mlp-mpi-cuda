#ifndef EXAMPLESETADAPTER_H_
#define EXAMPLESETADAPTER_H_

#include "mlp/ExampleSet.h"
#include "database/RelationAdapter.h"

using namespace MLP;

namespace Database
{

/**
 * Classe responsável por realizar operações sobre um conjunto de dados na base
 */
class ExampleSetAdapter
{

public:

	/**
	 * Recupera um conjunto de entrada
	 * @param set Conjunto de entrada a ser selecionado
	 * @param mlpID ID da rede
	 */
	static void select(ExampleSet &set, int mlpID);

private:

	/**
	 * Prepara a conexão para operações de seleção
	 * @param conn Conexão
	 */
	static void prepareForSelect(connection* conn);

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
	 * @param mlpID ID da rede
	 * @param set Conjunto de dados
	 * @return ID da relação de treinamento
	 */
	static int selectTrainedRelation(int mlpID, ExampleSet &set,
			WorkPtr &work);

	/**
	 * Seleciona o intervalo de valores do MLP
	 * @param mlpID ID da rede
	 * @param work Trabalho
	 * @return Intervalo de valores do MLP
	 */
	static Range selectRange(int mlpID, WorkPtr &work);

	/**
	 * Seleciona as estatísticas de uma relação
	 * @param mlpID ID da rede
	 * @param set Conjunto de dados a serem preenchidos
	 * @param nattr Quantidade de atributos da relação
	 * @param work Trabalho
	 */
	static void selectStatistics(int mlpID, ExampleSet &set, WorkPtr &work);

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

};

}

#endif
