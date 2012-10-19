#include "arff/Relation.h"
#include "arff/Scanner.h"

namespace ParallelMLP
{

//===========================================================================//

Relation::Relation(Driver &cDriver)
		: driver(cDriver)
{

}

//===========================================================================//

Relation::~Relation()
{
	// Deleta os atributos
	for (Attribute* attr : attributes)
		delete attr;

	// Deleta as instâncias
	for (Instance* inst : data)
	{
		for (Value* val : *inst)
			delete val;
		delete inst;
	}
}

//===========================================================================//

void Relation::addAttribute(Attribute* attr)
{
	// Verifica o tipo do atributo (só aceita numérico ou nominal)
	if (!attr->isNumeric() && !attr->isNominal())
		throwError(SEM_TYPE_NOT_ALLOWED);

	// Verifica se o nome do atributo já não foi utilizado anteriormente
	if (attrMap.find(attr->getName()) != attrMap.end())
		throwError(SEM_SAME_ATTRIBUTE_NAME);

	// Se o tipo for nominal
	if (attr->isNominal())
	{
		// Verifica se existem valores nominais repetidos
		map<string, bool> nominalMap;
		for (const string &str : attr->getNominalList())
		{
			if (nominalMap.find(str) != nominalMap.end())
				throwError(SEM_SAME_NOMINAL_VALUE);
			nominalMap[str] = true;
		}
	}

	attrMap[attr->getName()] = true;

	if (!attributes.empty())
		attributes.back()->setLast(false);

	attributes.push_back(attr);
	attributes.back()->setLast(true);
}

//===========================================================================//

void Relation::addInstance(const DataList* dlist, bool isSparse)
{
	Instance* inst;

	// Se não for esparso
	if (!isSparse)
		inst = new Instance(dlist->begin(), dlist->end());

	// Caso for esparso
	else
	{
		inst = new Instance();

		// Para cada valor da lista
		auto it = dlist->begin();
		for (uint i = 0; i < attributes.size(); i++)
		{
			if (it != dlist->end() && (*it)->getIndex() == i)
				inst->push_back(*it), it++;
			else
				inst->push_back(new Value(EMPTY));
		}

		// Caso os índices da lista esparsa estivem errados
		if (it != dlist->end())
			throwError(SEM_WRONG_INSTANCE_TYPE);
	}

	// Verifica a quantidade de valores
	if (inst->size() != attributes.size())
		throwError(SEM_WRONG_INSTANCE_TYPE);

	// Verifica os tipos de cada valor
	for (uint i = 0; i < inst->size(); i++)
	{
		Value* value = inst->at(i);

		if (!value->isEmpty() && value->getType() != attributes[i]->getType())
			throwError(SEM_WRONG_INSTANCE_TYPE);

		// Se for nominal, checa se o valor foi declarado
		if (value->isNominal())
		{
			int nominalIndex = checkNominal(i, *(value->str));

			// Se foi, seta o índice do valor nominal
			if (nominalIndex != -1)
				value->nominal = nominalIndex;
			else
				throwError(SEM_NOMINAL_NOT_DECLARED);
		}
	}

	inst->back()->setLast(true);
	data.push_back(inst);
}

//===========================================================================//

int Relation::checkNominal(uint attrIndex, const string &name)
{
	const Nominal& nominal = attributes[attrIndex]->getNominalList();
	uint i = 1;

	// Para cada valor nominal do atributo
	for (auto it = nominal.begin(); it != nominal.end(); it++, i++)
		// Se for igual, retorna o índice
		if (!name.compare(*it))
			return i;

	// Se não encontrar, retorna -1
	return -1;
}

//===========================================================================//

uint Relation::getNAttributes() const
{
	return attributes.size();
}

//===========================================================================//

uint Relation::getNInstances() const
{
	return data.size();
}

//===========================================================================//

string Relation::getName() const
{
	return name;
}

//===========================================================================//

void Relation::setName(const string *name)
{
	this->name = *name;
}

//===========================================================================//

const Attributes& Relation::getAttributes() const
{
	return attributes;
}

//===========================================================================//

const Attribute& Relation::getAttribute(uint i) const
{
	return *(attributes[i]);
}

//===========================================================================//

const Data& Relation::getData() const
{
	return data;
}

//===========================================================================//

const Instance& Relation::getInstance(uint i) const
{
	return *(data[i]);
}

//===========================================================================//

void Relation::throwError(ErrorType error) const
{
	throw ParallelMLPException(error, driver.scanner->getToken(),
			driver.scanner->getLineno());
}

//===========================================================================//

}
