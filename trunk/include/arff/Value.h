#ifndef VALUE_H_
#define VALUE_H_

#include "arff/Attribute.h"

namespace ARFF
{

/**
 * Classe que representa o valor de um dado
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
