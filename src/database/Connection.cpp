#include "database/Connection.h"
#include "database/DatabaseException.h"

namespace ParallelMLP
{

//===========================================================================//

Connection::Connection()
{
	try
	{
		conn = new connection(
				"dbname=mlpdb host=localhost user=mlpuser password=mlpuser");
	}
	catch (pqxx_exception &e)
	{
		conn = NULL;
		throw DatabaseException(COULD_NOT_CONNECT);
	}
}

//===========================================================================//

Connection::~Connection()
{
	if (conn != NULL)
	{
		conn->disconnect();
		delete conn;
	}
}

//===========================================================================//

connection* Connection::get()
{
	return conn;
}

//===========================================================================//

WorkPtr Connection::getWork() const
{
	return WorkPtr(new work(*conn));
}

//===========================================================================//

}
