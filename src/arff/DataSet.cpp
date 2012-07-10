#include "arff/DataSet.h"

namespace ARFF
{

/**
 * Constrói um conjunto de dados vazio
 */
DataSet::DataSet(Driver &cDriver)
    : driver(cDriver)
{

}

/**
 * Destrói o conjunto de dados
 */
DataSet::~DataSet()
{

}

/**
 * Seta o nome da relação
 * @param name Nome da relação
 */
void DataSet::setRelation(const string *name)
{
    relation = *name;
}

/**
 * Adiciona um atributo
 * @param attr Atributo
 */
void DataSet::addAttribute(Attribute *attr)
{
    AttributePtr ptr(attr);

    // Verifica o tipo do atributo (só aceita numérico ou nominal)
    if(ptr->type != NUMERIC && ptr->type != NOMINAL)
        throwError(SEM_TYPE_NOT_ALLOWED);

    attributes.push_back(ptr);
}

/**
 * Adiciona uma instância de dados
 * @param dlist Lista de valores da instância
 * @param isSparse Verdadeiro se a lista for esparsa ou falso caso contrário
 */
void DataSet::addInstance(const DataList* dlist, bool isSparse)
{
    InstancePtr row;

    // Se não for esparso
    if(!isSparse)
        row = InstancePtr(new Instance(dlist->begin(), dlist->end()));

    // Caso for esparso
    else
    {
        row = InstancePtr(new Instance());

        // Para cada valor da lista
        for(ValuePtr value : *dlist)
        {
            // Adicona valores vazios
            for(uint i = row->size(); i < value->index; i++)
                row->push_back(ValuePtr(new Value(EMPTY)));
            row->push_back(value);
        }
    }

    // Verifica a quantidade de valores
    if(row->size() != attributes.size())
        throwError(SEM_INVALID_TYPE);

    // Verifica os tipos de cada valor
    for(uint i = 0; i < row->size(); i++)
    {
        ValuePtr &value = row->at(i);
        if(value->type != EMPTY && value->type != attributes[i]->type)
            throwError(SEM_INVALID_TYPE);
    }

    data.push_back(row);
}

}
