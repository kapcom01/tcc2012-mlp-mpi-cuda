#include "arff/Value.h"

namespace ParallelMLP
{

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
