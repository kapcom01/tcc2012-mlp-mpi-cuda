#include "arff/DataSet.h"
#include "arff/ParseException.h"

namespace ARFF
{

//===========================================================================//

DataSet::DataSet(Driver &cDriver)
		: driver(cDriver)
{

}

//===========================================================================//

DataSet::~DataSet()
{

}

//===========================================================================//

void DataSet::setRelation(const string *name)
{
	relation = *name;
}

//===========================================================================//

void DataSet::addAttribute(Attribute *attr)
{
	AttributePtr ptr(attr);

	// Verifica o tipo do atributo (só aceita numérico ou nominal)
	if (ptr->type != NUMERIC && ptr->type != NOMINAL)
		throwError(SEM_TYPE_NOT_ALLOWED);

	// Verifica se o nome do atributo já não foi utilizado anteriormente
	if (attrMap.find(attr->name) != attrMap.end())
		throwError(SEM_SAME_ATTRIBUTE_NAME);

	// Se o tipo for nominal
	if (attr->type == NOMINAL)
	{
		// Verifica se existem valores nominais repetidos
		map<string, bool> nominalMap;
		for (string &str : *(attr->nominal))
		{
			if (nominalMap.find(str) != nominalMap.end())
				throwError(SEM_SAME_NOMINAL_VALUE);
			nominalMap[str] = true;
		}
	}

	attrMap[attr->name] = true;
	attributes.push_back(ptr);
}

//===========================================================================//

void DataSet::addInstance(const DataList* dlist, bool isSparse)
{
	InstancePtr row;

	// Se não for esparso
	if (!isSparse)
		row = InstancePtr(new Instance(dlist->begin(), dlist->end()));

	// Caso for esparso
	else
	{
		row = InstancePtr(new Instance());

		// Para cada valor da lista
		auto it = dlist->begin();
		for (uint i = 0; i < attributes.size(); i++)
		{
			if (it != dlist->end() && (*it)->index == i)
				row->push_back(*it), it++;
			else
				row->push_back(ValuePtr(new Value(EMPTY)));
		}

		// Caso os índices da lista esparsa estivem errados
		if (it != dlist->end())
			throwError(SEM_WRONG_INSTANCE_TYPE);
	}

	// Verifica a quantidade de valores
	if (row->size() != attributes.size())
		throwError(SEM_WRONG_INSTANCE_TYPE);

	// Verifica os tipos de cada valor
	for (uint i = 0; i < row->size(); i++)
	{
		ValuePtr &value = row->at(i);
		if (value->type != EMPTY && value->type != attributes[i]->type)
			throwError(SEM_WRONG_INSTANCE_TYPE);
	}

	data.push_back(row);
}

//===========================================================================//

}
