#include "arff/Attribute.h"

namespace ParallelMLP
{

//===========================================================================//

Attribute::Attribute(const string &name, AttributeType type)
{
	this->name = name;
	this->type = type;
}

//===========================================================================//

Attribute::Attribute(const string &name, AttributeType type,
		const string &format)
{
	this->name = name;
	this->type = type;
	this->str = new string(format);
}

//===========================================================================//

Attribute::Attribute(const string &name, AttributeType type,
		const Nominal &nominal)
{
	this->name = name;
	this->type = type;
	this->nominal = new Nominal(nominal.begin(), nominal.end());
}

//===========================================================================//

Attribute::~Attribute()
{
	if (type == DATE)
		delete str;
	else if (type == NOMINAL)
		delete nominal;
}

//===========================================================================//

bool Attribute::isNominal() const
{
	return (type == NOMINAL);
}

//===========================================================================//

bool Attribute::isNumeric() const
{
	return (type == NUMERIC);
}

//===========================================================================//

string Attribute::getName() const
{
	return name;
}

//===========================================================================//

AttributeType Attribute::getType() const
{
	return type;
}

//===========================================================================//

const Nominal& Attribute::getNominalList() const
{
	return *nominal;
}

//===========================================================================//

uint Attribute::getNominalCard() const
{
	return nominal->size();
}

//===========================================================================//

}
