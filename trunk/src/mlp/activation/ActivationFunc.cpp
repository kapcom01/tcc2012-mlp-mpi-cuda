#include "mlp/activation/ActivationFunc.h"

namespace MLP
{

//===========================================================================//

ActivationFunc::ActivationFunc(int type)
{
	this->type = type;
}

//===========================================================================//

ActivationFunc::~ActivationFunc()
{

}

//===========================================================================//

int ActivationFunc::getType()
{
	return type;
}

//===========================================================================//

}
