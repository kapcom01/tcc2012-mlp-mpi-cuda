#ifndef DATASETTYPES_H_
#define DATASETTYPES_H_

#include "Common.h"
#include <list>

namespace ARFF
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
 * Atributo
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

/**
 * Valor de um dado
 */
class Value
{

public:

	/**
	 * Constrói um valor
	 * @param type Tipo do atributo
	 */
	Value(AttributeType type);

	/**
	 * Constrói um valor
	 * @param type Tipo do atributo
	 * @param number Valor numérico
	 */
	Value(AttributeType type, double number);

	/**
	 * Constrói um valor
	 * @param type Tipo do atributo
	 * @param str Valor nominal ou string
	 */
	Value(AttributeType type, string &str);

	/**
	 * Destrói o valor
	 */
	virtual ~Value();

	/**
	 * Índice
	 */
	int index;

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
		 * Valor numérico
		 */
		double number;

		/**
		 * Valor string
		 */
		string* str;
	};
};

/**
 * Ponteiro para DataValue
 */
typedef shared_ptr<Value> ValuePtr;

/**
 * Dados de uma linha
 */
typedef vector<ValuePtr> Instance;

/**
 * Dados de uma linha como uma lista
 */
typedef list<ValuePtr> DataList;

/**
 * Ponteiro para DataRow
 */
typedef shared_ptr<Instance> InstancePtr;

/**
 * Vários dados
 */
typedef vector<InstancePtr> Data;

}

#endif
