#ifndef RELATION_H_
#define RELATION_H_

#include "arff/Value.h"
#include "arff/Driver.h"
#include <map>

namespace ParallelMLP
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
	 * Retorna a quantidade de atributos
	 * @return Quantidade de atributos
	 */
	uint getNAttributes() const;

	/**
	 * Retorna a quantidade de instâncias
	 * @return Quantidade de instâncias
	 */
	uint getNInstances() const;

	/**
	 * Retorna o nome da relação
	 * @return Nome da relação
	 */
	string getName() const;

	/**
	 * Seta o nome da relação
	 * @param name Nome da relação
	 */
	void setName(const string *name);

	/**
	 * Retorna o i-ésimo atributo
	 * @param i Índice do atributo
	 * @return i-ésimo atributo
	 */
	const Attribute& getAttribute(uint i) const;

	/**
	 * Retorna a i-ésima instância
	 * @param i Índice da instância
	 * @return i-ésima instância
	 */
	const Instance& getInstance(uint i) const;

private:

	/**
	 * Verifica se um valor nominal foi declarado
	 * @param attrIndex Índice do atributo
	 * @param name Valor nominal
	 * @return Índice do valor nominal
	 */
	int checkNominal(uint attrIndex, const string &name);

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
