#ifndef DATASET_H_
#define DATASET_H_

#include "arff/DataTypes.h"
#include "arff/Driver.h"
#include <map>

namespace ARFF
{

/**
 * Conjunto de dados extraídos do arquivo ARFF
 */
class DataSet
{

public:

	/**
	 * Constrói um conjunto de dados vazio
	 */
	DataSet(Driver &driver);

	/**
	 * Destrói o conjunto de dados
	 */
	~DataSet();

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

	/**
	 * Nome da relação
	 */
	string relation;

	/**
	 * Informações sobre os atributos
	 */
	Attributes attributes;

	/**
	 * Informações sobre os dados
	 */
	Data data;

private:

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
typedef shared_ptr<DataSet> DataSetPtr;

}

#endif
