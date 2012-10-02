#include "exception/ParallelMLPException.h"

namespace ParallelMLP
{

//===========================================================================//

map<int, string> ParallelMLPException::messages =
{
	// Erros de parseamento de arquivos ARFF
	{ LEX_MALFORMED_NUMBER,
			"number '%s' malformed" },
	{ LEX_QUOTATION_NOT_CLOSED,
			"quotation marks on '%s' not closed" },
	{ LEX_UNKNOWN_TOKEN,
			"unknown token '%s'" },
	{ SIN_INVALID_RELATION,
			"invalid relation declaration, the correct format is: @relation "
			"<relation-name>" },
	{ SIN_INVALID_ATTRIBUTE,
			"invalid attribute declaration, the correct format is: @attribute "
			"<attribute-name> <datatype>" },
	{ SIN_INVALID_INSTANCE,
			"invalid instance, the correct format is: <value>,<value>,...,"
			"<value>" },
	{ SEM_TYPE_NOT_ALLOWED,
			"attribute type not allowed, it should be numeric or nominal" },
	{ SEM_WRONG_INSTANCE_TYPE,
			"types of this instance do not correspond to the types of the "
			"declared attributes" },
	{ SEM_SAME_ATTRIBUTE_NAME,
			"attribute name has previously declared" },
	{ SEM_SAME_NOMINAL_VALUE,
			"nominal value has previously declared" },
	{ SEM_NOMINAL_NOT_DECLARED,
			"nominal value not declared before"	},

	// Erros de operações em bases de dados
	{ COULD_NOT_CONNECT,
			"could not connect to database" },
	{ RELATION_NOT_UNIQUE,
			"relation name already exists, choose another" },

	// Erros de operações de um MLP
	{ INVALID_INPUT_VARS,
			"invalid example set, number of input vars does not match" },
	{ INVALID_OUTPUT_VARS,
			"invalid example set, number of output vars does not match" }
};

//===========================================================================//

ParallelMLPException::ParallelMLPException(ErrorType error)
{
	this->error = error;
	msg = "ERROR:  " + messages[error];
}

//===========================================================================//

ParallelMLPException::ParallelMLPException(ErrorType error, string info)
{
	this->error = error;

	char out[500];
	sprintf(out, messages[error].c_str(), info.c_str());
	msg = "ERROR:  " + string(out);
}

//===========================================================================//

ParallelMLPException::ParallelMLPException(ErrorType error, string info,
		int lineno)
{
	this->error = error;

	char out[500];
	sprintf(out, messages[error].c_str(), info.c_str());
	msg = "ERROR:  line " + to_string(lineno) + ": " + out;
}

//===========================================================================//

ParallelMLPException::ParallelMLPException(const exception& ex)
{
	this->error = OTHER_ERRORS;
	msg = ex.what();
}

//===========================================================================//

ParallelMLPException::~ParallelMLPException() throw ()
{

}

//===========================================================================//

const char* ParallelMLPException::what() const throw ()
{
	return msg.c_str();
}

//===========================================================================//

ErrorType ParallelMLPException::getErrorType() const
{
	return error;
}

//===========================================================================//

}
