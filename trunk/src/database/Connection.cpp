#include "database/Connection.h"

namespace ParallelMLP
{

//===========================================================================//

ConnectionPtr Connection::conn = ConnectionPtr(new Connection());

//===========================================================================//

Connection::Connection()
{
	try
	{
		baseConn = new connection(
				"dbname=mlpdb host=localhost user=mlpuser password=mlpuser");
	}
	catch (pqxx_exception &e)
	{
		baseConn = NULL;
		throw ParallelMLPException(COULD_NOT_CONNECT);
	}
}

//===========================================================================//

Connection::~Connection()
{
	if (baseConn != NULL)
	{
		baseConn->disconnect();
		delete baseConn;
	}
}

//===========================================================================//

connection& Connection::get()
{
	if (conn.get() == NULL)
		throw ParallelMLPException(COULD_NOT_CONNECT);
	else
		return *(conn->baseConn);
}

//===========================================================================//

}
