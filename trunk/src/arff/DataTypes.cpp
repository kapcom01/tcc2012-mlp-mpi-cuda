#include "arff/DataTypes.h"

namespace ARFF
{

//===========================================================================//

Attribute::Attribute(const string &name, AttributeType type)
{
	this->name = name;
	this->type = type;
}

//===========================================================================//

Attribute::Attribute(const string &name, AttributeType type, const string &format)
{
	this->name = name;
	this->type = type;
	this->str = new string(format);
}

//===========================================================================//

Attribute::Attribute(const string &name, AttributeType type, const Nominal &nominal)
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


Value::Value(AttributeType type)
{
	this->type = type;
}

//===========================================================================//

Value::Value(AttributeType type, double number)
{
	this->type = type;
	this->number = number;
}

//===========================================================================//

Value::Value(AttributeType type, string &str)
{
	this->type = type;
	this->number = number;
	this->str = new string(str);
}

//===========================================================================//

Value::~Value()
{
	if (type == STRING || type == NOMINAL || type == DATE)
		delete str;
}

//===========================================================================//

}
