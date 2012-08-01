#ifndef INSERTER_H_
#define INSERTER_H_

#include "arff/DataSet.h"
#include "database/Connection.h"

using namespace ARFF;

namespace Database
{

/**
 * Classe responsável por inserir um conjunto de dados na base
 */
class Inserter
{

public:

	/**
	 * Insere um conjunto de dados
	 * @param dataset Conjunto de dados
	 */
	static void insert(const DataSet &dataset);

private:

	/**
	 * Verifica se o nome da relação já existe anteriormente
	 * @param name Nome da relação
	 * @param work Trabalho
	 * @return Verdadeiro se não existir ou falso caso contrário
	 */
	static bool checkUnique(const string &name, WorkPtr &work);

	/**
	 * Insere as informações da relação
	 * @param dataset Conjunto de dados
	 * @param work Trabalho
	 * @return ID gerado
	 */
	static int insertRelation(const DataSet &dataset, WorkPtr &work);

	/**
	 * Insere um atributo
	 * @param relationID Identificador da relação
	 * @param attrIndex Índice do atributo
	 * @param attr Atributo
	 * @param work Trabalho
	 */
	static void insertAttribute(uint relationID, uint attrIndex,
			const Attribute &attr, WorkPtr &work);

	/**
	 * Insere uma instância
	 * @param relationID Identificador da relação
	 * @param instIndex Índice da instância
	 * @param inst Instância
	 * @param work Trabalho
	 */
	static void insertInstance(uint relationID, uint instIndex,
			const Instance &inst, WorkPtr &work);

};

}

#endif
