#include "arff/ParseException.h"
#include "arff/Scanner.h"

namespace ParallelMLP
{

/**
* Inicializa as mensagens de erro
*/
unordered_map<int, string> ParseException::messages =
{
	{ LEX_MALFORMED_NUMBER,
			"number '%s' malformed" },
	{ LEX_QUOTATION_NOT_CLOSED,
			"quotation marks on '%s' not closed" },
	{ LEX_UNKNOWN_TOKEN,
			"unknown token '%s'" },
	{ SIN_INVALID_RELATION,
			"invalid relation declaration, the correct format is: @relation <relation-name>" },
	{ SIN_INVALID_ATTRIBUTE,
			"invalid attribute declaration, the correct format is: @attribute <attribute-name> <datatype>" },
	{ SIN_INVALID_INSTANCE,
			"invalid instance, the correct format is: <value>,<value>,...,<value>" },
	{ SEM_TYPE_NOT_ALLOWED,
			"attribute type not allowed, it should be numeric or nominal" },
	{ SEM_WRONG_INSTANCE_TYPE,
			"types of this instance do not correspond to the types of the declared attributes" },
	{ SEM_SAME_ATTRIBUTE_NAME,
			"attribute name has previously declared" },
	{ SEM_SAME_NOMINAL_VALUE,
			"nominal value has previously declared" },
	{ SEM_NOMINAL_NOT_DECLARED,
			"nominal value not declared before"	}
};

//===========================================================================//

ParseException::ParseException(int error, Driver &driver)
{
	char out[500];
	sprintf(out, messages[error].c_str(), driver.scanner->getToken().c_str());
	msg = "ERROR:  line " + to_string(driver.scanner->getLineno()) + ": " + out;
}

//===========================================================================//

ParseException::~ParseException() throw ()
{

}

//===========================================================================//

const char* ParseException::what() const throw ()
{
	return msg.c_str();
}

//===========================================================================//

}
