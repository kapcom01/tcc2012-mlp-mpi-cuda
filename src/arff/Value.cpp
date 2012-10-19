#include "arff/Value.h"

namespace ParallelMLP
{

//===========================================================================//


Value::Value(AttributeType type)
{
	this->index = -1;
	this->type = type;
	this->last = false;
}

//===========================================================================//

Value::Value(AttributeType type, float number)
{
	this->index = -1;
	this->type = type;
	this->number = number;
	this->last = false;
}

//===========================================================================//

Value::Value(AttributeType type, string &str)
{
	this->index = -1;
	this->type = type;
	this->number = number;
	this->str = new string(str);
	this->last = false;
}

//===========================================================================//

Value::~Value()
{
	if (type == STRING || type == NOMINAL || type == DATE)
		delete str;
}

//===========================================================================//

bool Value::isNumeric() const
{
	return (type == NUMERIC);
}

//===========================================================================//

bool Value::isNominal() const
{
	return (type == NOMINAL);
}

//===========================================================================//

bool Value::isEmpty() const
{
	return (type == EMPTY);
}

//===========================================================================//

int Value::getIndex() const
{
	return index;
}

//===========================================================================//

void Value::setIndex(int index)
{
	this->index = index;
}

//===========================================================================//

AttributeType Value::getType() const
{
	return type;
}

//===========================================================================//

bool Value::isLast() const
{
	return last;
}

//===========================================================================//

void Value::setLast(bool last)
{
	this->last = last;
}

//===========================================================================//

float Value::getNumber() const
{
	return number;
}

//===========================================================================//

int Value::getNominal() const
{
	return nominal;
}

//===========================================================================//

}
