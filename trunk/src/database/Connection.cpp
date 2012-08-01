#include "database/Connection.h"
#include "database/DatabaseException.h"

namespace Database
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

	try
	{
		// Verificação de unicidade da relação
		conn->prepare("checkUnique",
				"SELECT COUNT(*) FROM Relation WHERE Name = $1")
				("VARCHAR", prepare::treat_string);

		// Inserção na tabela Relation
		conn->prepare("insertRelation",
				"INSERT INTO Relation (Name, NAttributes, NInstances) VALUES ($1, $2, $3)")
				("VARCHAR", prepare::treat_string)("INTEGER")("INTEGER");

		// Inserção na tabela Attribute
		conn->prepare("insertAttribute",
				"INSERT INTO Attribute (RelationID, AttrIndex, Name, Type, NominalCard) VALUES ($1, $2, $3, $4, $5)")
				("INTEGER")("INTEGER")("VARCHAR", prepare::treat_string)
				("SMALLINT")("INTEGER");

		// Inserção na tabela Nominal
		conn->prepare("insertNominal",
				"INSERT INTO Nominal (RelationID, AttrIndex, Name) VALUES ($1, $2, $3)")
				("INTEGER")("INTEGER")("VARCHAR", prepare::treat_string);

		// Inserção na tabela Instance
		conn->prepare("insertInstance",
				"INSERT INTO Data (RelationID, InstanceIndex, AttrIndex, NumericValue, NominalValue) VALUES ($1, $2, $3, $4, $5)")
				("INTEGER")("INTEGER")("INTEGER")("NUMERIC")
				("VARCHAR",	prepare::treat_string);

		// Seleção do último ID de Relation gerado
		conn->prepare("selectLastID",
				"SELECT currval(pg_get_serial_sequence('Relation', 'relationid'))");
	}
	catch (pqxx_exception &e)
	{
		throw DatabaseException(e);
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

WorkPtr Connection::getWork() const
{
	return WorkPtr(new work(*conn));
}

//===========================================================================//

}
