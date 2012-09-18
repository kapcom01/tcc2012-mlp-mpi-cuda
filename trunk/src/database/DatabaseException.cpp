#include "database/DatabaseException.h"

namespace ParallelMLP
{

/**
 * Inicializa as mensagens de erro
 */
unordered_map<int, string> DatabaseException::messages =
{
	{ COULD_NOT_CONNECT,
			"could not connect to database" },
	{ RELATION_NOT_UNIQUE,
			"relation name already exists, choose another" }
};

//===========================================================================//

DatabaseException::DatabaseException(int error)
{
	msg = "ERROR:  " + messages[error];
}

//===========================================================================//

DatabaseException::DatabaseException(pqxx_exception &error)
{
	msg = error.base().what();
}

//===========================================================================//

DatabaseException::~DatabaseException() throw ()
{

}

//===========================================================================//

const char* DatabaseException::what() const throw ()
{
	return msg.c_str();
}

//===========================================================================//

}
