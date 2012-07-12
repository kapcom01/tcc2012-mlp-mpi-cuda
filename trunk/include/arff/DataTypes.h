#ifndef DATASETTYPES_H_
#define DATASETTYPES_H_

#include "Defines.h"
#include <list>
#include <vector>

namespace ARFF
{

/**
 * Tipo nominal
 */
typedef list<string> Nominal;

/**
 * Tipos de atributo
 */
enum AttributeType { NUMERIC, NOMINAL, STRING, DATE, EMPTY };

/**
 * Atributo
 */
struct Attribute
{
    string name;          // Nome
    AttributeType type;   // Tipo

    union
    {
        string* str;      // Formato da data
        Nominal* nominal; // Valores nominais
    };

    /**
     * Constrói um novo atributo
     * @param name Nome do atributo
     * @param type Tipo do atributo
     */
    Attribute(const string &name, AttributeType type)
    {
        this->name = name;
        this->type = type;
    }

    /**
     * Constrói um novo atributo
     * @param name Nome do atributo
     * @param type Tipo do atributo
     * @param format Formato da data
     */
    Attribute(const string &name, AttributeType type, const string &format)
    {
        this->name = name;
        this->type = type;
        this->str = new string(format);
    }

    /**
     * Constrói um novo atributo
     * @param name Nome do atributo
     * @param type Tipo do atributo
     * @param nominal Lista de atributos nominais
     */
    Attribute(const string &name, AttributeType type, const Nominal &nominal)
    {
        this->name = name;
        this->type = type;
        this->nominal = new Nominal(nominal.begin(), nominal.end());
    }

    /**
     * Destrói o atributo
     */
    virtual ~Attribute()
    {
        if(type == DATE)
            delete str;
        else if(type == NOMINAL)
            delete nominal;
    }
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
struct Value
{
    int index;            // Índice
    AttributeType type;   // Tipo

    union
    {
        double number;    // Valor numérico
        string* str;      // Valor string
    };

    /**
     * Constrói um valor
     * @param type Tipo do atributo
     */
    Value(AttributeType type)
    {
        this->type = type;
    }

    /**
     * Constrói um valor
     * @param type Tipo do atributo
     * @param number Valor numérico
     */
    Value(AttributeType type, double number)
    {
        this->type = type;
        this->number = number;
    }

    /**
     * Constrói um valor
     * @param type Tipo do atributo
     * @param str Valor nominal ou string
     */
    Value(AttributeType type, string &str)
    {
        this->type = type;
        this->number = number;
        this->str = new string(str);
    }

    /**
     * Destrói o valor
     */
    virtual ~Value()
    {
        if(type == STRING || type == NOMINAL || type == DATE)
            delete str;
    }
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
