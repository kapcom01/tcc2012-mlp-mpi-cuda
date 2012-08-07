#ifndef RELATION_H_
#define RELATION_H_

#include "arff/Value.h"
#include "arff/Driver.h"
#include <map>

namespace Database { class RelationHelper; }

namespace ARFF
{

/**
 * Conjunto de dados extraídos do arquivo ARFF
 */
class Relation
{

public:

	/**
	 * Constrói um conjunto de dados vazio
	 */
	Relation(Driver &driver);

	/**
	 * Destrói o conjunto de dados
	 */
	~Relation();

	/**
	 * Seta o nome da relação
	 * @param name Nome da relação
	 */
	void setRelation(const string *name);

	/**
	 * Adiciona um atributo
	 * @param attr Atributo
	 */
	void addAttribute(Attribute *attr);

	/**
	 * Adiciona uma instância de dados
	 * @param dlist Lista de valores da instância
	 * @param isSparse Verdadeiro se a lista for esparsa ou falso caso contrário
	 */
	void addInstance(const DataList* dlist, bool isSparse);

	friend class Database::RelationHelper;

private:

	/**
	 * Nome da relação
	 */
	string name;

	/**
	 * Informações sobre os atributos
	 */
	Attributes attributes;

	/**
	 * Informações sobre os dados
	 */
	Data data;

	/**
	 * Driver
	 */
	Driver &driver;

	/**
	 * Mapeamento dos nomes dos atributos
	 */
	map<string, bool> attrMap;

};

/**
 * Ponteiro inteligente para Dataset
 */
typedef shared_ptr<Relation> DataSetPtr;

}

#endif
