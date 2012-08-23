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
	 * @param relationID ID da relação
	 * @param inputSet Conjunto de entrada a ser selecionado
	 */
	static void select(int relationID, ExampleSet &inputSet);

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
	 * @param relationID Identificador da relação
	 * @param inputSet Conjunto de dados a serem preenchidos
	 * @param nattr Quantidade de atributos da relação
	 * @param work Trabalho
	 */
	static void selectData(int relationID, ExampleSet &inputSet, int nattr,
			WorkPtr &work);

};

}

#endif
