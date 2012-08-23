#include "database/ExampleSetAdapter.h"
#include "database/DatabaseException.h"

namespace Database
{

//===========================================================================//

void ExampleSetAdapter::prepareForSelect(connection* conn)
{
	try
	{
		// Seleção da quantidade de atributos de uma relação
		conn->prepare("selectNAttr",
				"SELECT NAttributes FROM Relation WHERE RelationID = $1")
				("INTEGER");

		// Selação dos dados da relação
		conn->prepare("selectData",
				"SELECT D.InstanceIndex, D.AttrIndex, A.Type, A.NominalCard, "
				"D.NumericValue, D.NominalValue FROM Attribute A, Data D "
				"WHERE A.AttrIndex = D.AttrIndex AND "
				"A.RelationID = D.RelationID AND A.RelationID = $1 "
				"ORDER BY D.InstanceIndex, D.AttrIndex")
				("INTEGER");
	}
	catch (pqxx_exception &e)
	{
		throw DatabaseException(e);
	}
}

//===========================================================================//

void ExampleSetAdapter::select(int relationID, ExampleSet &inputSet)
{
	// Cria uma nova conexão com a base de dados
	Connection conn;
	prepareForSelect(conn.get());

	WorkPtr work = conn.getWork();

	try
	{
		int nattr = selectNAttributes(relationID, work);
		selectData(relationID, inputSet, nattr, work);
	}
	catch (pqxx_exception &e)
	{
		// Desfaz as alterações
		work->abort();
		throw DatabaseException(e);
	}
}

//===========================================================================//

int ExampleSetAdapter::selectNAttributes(int relationID, WorkPtr &work)
{
	const result &res = work->prepared("selectNAttr")(relationID).exec();
	return res[0][0].as<int>();
}

//===========================================================================//

void ExampleSetAdapter::selectData(int relationID, ExampleSet &inputSet,
		int nattr, WorkPtr &work)
{
	const result &res = work->prepared("selectData")(relationID).exec();

	// Para cada tupla do resultado
	for (auto row = res.begin(); row != res.end(); row++)
	{
		int attrIndex = row["AttrIndex"].as<int>();

		// Adiciona uma nova instância
		if (attrIndex == 1)
			inputSet.pushInstance();

		// Se for numérico
		if (row["Type"].as<int>() == NUMERIC_TYPE)
		{
			double value = row["NumericValue"].as<double>();
			inputSet.addValue(value, attrIndex == nattr);
		}

		// Se for nominal
		else
		{
			double value = row["NominalValue"].as<int>();
			int card = row["NominalCard"].as<int>();
			inputSet.addValue(value, card, attrIndex == nattr);
		}
	}
}

//===========================================================================//

}
