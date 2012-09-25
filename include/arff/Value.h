#ifndef VALUE_H_
#define VALUE_H_

#include "arff/Attribute.h"

namespace ParallelMLP
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
	Value(AttributeType type, float number);

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
	 * Verifica se o valor é numérico
	 * @return Verdadeiro se o valor for numérico ou falso caso contrário
	 */
	bool isNumeric() const;

	/**
	 * Verifica se o valor é nominal
	 * @return Verdadeiro se o valor for nominal ou falso caso contrário
	 */
	bool isNominal() const;

	/**
	 * Verifica se o valor está vazio
	 * @return Verdadeiro se o valor for vazio ou falso caso contrário
	 */
	bool isEmpty() const;

	/**
	 * Retorna o índice associado ao valor
	 * @return Índice associado ao valor
	 */
	int getIndex() const;

	/**
	 * Seta o índice associado ao valor
	 * @param index Índice associado ao valor
	 */
	void setIndex(int index);

	/**
	 * Retorna o tipo do valor
	 * @return Tipo do valor
	 */
	AttributeType getType() const;

	/**
	 * Retorna o valor numérico
	 * @return Valor numérico
	 */
	float getNumber() const;

	/**
	 * Retorna o valor nominal
	 * @return Valor nominal
	 */
	int getNominal() const;

private:

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
		float number;

		struct
		{
			/**
			 * Valor string
			 */
			string* str;

			/**
			 * Valor nominal
			 */
			int nominal;
		};
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
