#ifndef ATTRIBUTE_H_
#define ATTRIBUTE_H_

#include "Common.h"
#include <list>

namespace ParallelMLP
{

/**
 * Tipo nominal
 */
typedef list<string> Nominal;

/**
 * Tipos de atributo
 */
enum AttributeType
{
	NUMERIC, NOMINAL, STRING, DATE, EMPTY
};

/**
 * Classe que representa um Atributo
 */
class Attribute
{

public:

	/**
	 * Constrói um novo atributo
	 * @param name Nome do atributo
	 * @param type Tipo do atributo
	 */
	Attribute(const string &name, AttributeType type);

	/**
	 * Constrói um novo atributo
	 * @param name Nome do atributo
	 * @param type Tipo do atributo
	 * @param format Formato da data
	 */
	Attribute(const string &name, AttributeType type, const string &format);

	/**
	 * Constrói um novo atributo
	 * @param name Nome do atributo
	 * @param type Tipo do atributo
	 * @param nominal Lista de atributos nominais
	 */
	Attribute(const string &name, AttributeType type, const Nominal &nominal);

	/**
	 * Destrói o atributo
	 */
	virtual ~Attribute();

	/**
	 * Verifica se o atributo é nominal
	 * @return Verdadeiro se for nominal ou falso caso contrário
	 */
	bool isNominal() const;

	/**
	 * Verifica se o atributo é numérico
	 * @return Verdadeiro se for numérico ou falso caso contrário
	 */
	bool isNumeric() const;

	/**
	 * Retorna o nome do atributo
	 * @return Nome do atributo
	 */
	string getName() const;

	/**
	 * Retorna o tipo do atributo
	 * @return Tipo do atributo
	 */
	AttributeType getType() const;

	/**
	 * Retorna uma lista de valores nominais
	 * @return Lista de valores nominais
	 */
	const Nominal& getNominalList() const;

private:

	/**
	 * Nome
	 */
	string name;

	/**
	 * Tipo
	 */
	AttributeType type;

	/**
	 * Valor
	 */
	union
	{
		/**
		 * Valor string
		 */
		string* str;

		/**
		 * Valores nominais
		 */
		Nominal* nominal;
	};
};

/**
 * Ponteiro para Attribute
 */
typedef shared_ptr<Attribute> AttributePtr;

/**
 * Vários atributos
 */
typedef vector<AttributePtr> Attributes;

}

#endif
