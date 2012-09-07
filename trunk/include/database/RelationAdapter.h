#ifndef RELATIONADAPTER_H_
#define RELATIONADAPTER_H_

#include "arff/Relation.h"
#include "database/Connection.h"

#define NUMERIC_TYPE 1
#define NOMINAL_TYPE 2

using namespace ARFF;

namespace Database
{

/**
 * Classe responsável por realizar operações sobre um conjunto de dados na base
 */
class RelationAdapter
{

public:

	/**
	 * Insere um conjunto de dados
	 * @param relation Conjunto de dados
	 */
	static void insert(const Relation &relation);

private:

	/**
	 * Prepara a conexão para operações de inserção
	 * @param conn Conexão
	 */
	static void prepareForInsert(connection* conn);

	/**
	 * Verifica se o nome da relação já existe anteriormente
	 * @param name Nome da relação
	 * @param work Trabalho
	 * @return Verdadeiro se não existir ou falso caso contrário
	 */
	static bool checkUnique(const string &name, WorkPtr &work);

	/**
	 * Insere as informações da relação
	 * @param relation Conjunto de dados
	 * @param work Trabalho
	 * @return ID gerado
	 */
	static int insertRelation(const Relation &relation, WorkPtr &work);

	/**
	 * Insere um atributo
	 * @param relationID Identificador da relação
	 * @param attrIndex Índice do atributo
	 * @param attr Atributo
	 * @param work Trabalho
	 */
	static void insertAttribute(int relationID, uint attrIndex,
			const Attribute &attr, WorkPtr &work);

	/**
	 * Insere uma instância
	 * @param relationID Identificador da relação
	 * @param instIndex Índice da instância
	 * @param inst Instância
	 * @param work Trabalho
	 */
	static void insertInstance(int relationID, uint instIndex,
			const Instance &inst, WorkPtr &work);

	/**
	 * Adiciona estatísticas sobre um atributo
	 * @param relationID Identificador da relação
	 * @param attrIndex Índice do atributo
	 * @param attr Atributo
	 * @param work Trabalho
	 */
	static void addStatistics(int relationID, uint attrIndex,
			const Attribute &attr, WorkPtr &work);

};

}

#endif
