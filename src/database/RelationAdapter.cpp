#include "database/RelationAdapter.h"
#include "database/DatabaseException.h"

namespace Database
{

//===========================================================================//

void RelationAdapter::prepareForInsert(connection* conn)
{
	try
	{
		// Verificação de unicidade da relação Relation
		conn->prepare("checkUnique",
				"SELECT COUNT(*) FROM Relation WHERE Name = $1")
				("VARCHAR", prepare::treat_string);

		// Inserção na tabela Relation
		conn->prepare("insertRelation",
				"INSERT INTO Relation (Name, NAttributes, NInstances) "
				"VALUES ($1, $2, $3)")
				("VARCHAR", prepare::treat_string)("INTEGER")("INTEGER");

		// Inserção na tabela Attribute
		conn->prepare("insertAttribute",
				"INSERT INTO Attribute (RelationID, AttrIndex, Name, Type, "
				"NominalCard) VALUES ($1, $2, $3, $4, $5)")
				("INTEGER")("INTEGER")("VARCHAR", prepare::treat_string)
				("SMALLINT")("INTEGER");

		// Inserção na tabela Nominal
		conn->prepare("insertNominal",
				"INSERT INTO Nominal (RelationID, AttrIndex, NominalIndex, "
				"Name) VALUES ($1, $2, $3, $4)")
				("INTEGER")("INTEGER")("SMALLINT")
				("VARCHAR", prepare::treat_string);

		// Inserção na tabela Instance
		conn->prepare("insertInstance",
				"INSERT INTO Data (RelationID, InstanceIndex, AttrIndex, "
				"NumericValue, NominalValue) VALUES ($1, $2, $3, $4, $5)")
				("INTEGER")("INTEGER")("INTEGER")("NUMERIC")("SMALLINT");

		// Seleção do último ID de Relation gerado
		conn->prepare("selectLastID",
				"SELECT currval(pg_get_serial_sequence('Relation', "
				"'relationid'))");
	}
	catch (pqxx_exception &e)
	{
		throw DatabaseException(e);
	}
}

//===========================================================================//

void RelationAdapter::insert(const Relation &relation)
{
	// Cria uma nova conexão com a base de dados
	Connection conn;
	prepareForInsert(conn.get());

	WorkPtr work = conn.getWork();

	try
	{
		// Verifica se o nome da relação é realmente único
		if (!checkUnique(relation.name, work))
			throw DatabaseException(RELATION_NOT_UNIQUE);

		// Insere a relação
		int relationID = insertRelation(relation, work);

		// Insere os atributos
		for (uint i = 0; i < relation.attributes.size(); i++)
			insertAttribute(relationID, i + 1, *(relation.attributes[i]), work);

		// Insere os dados
		for (uint i = 0; i < relation.data.size(); i++)
			insertInstance(relationID, i + 1, *(relation.data[i]), work);

		// Salva as alterações
		work->commit();
	}
	catch (pqxx_exception &e)
	{
		// Desfaz as alterações
		work->abort();
		throw DatabaseException(e);
	}
}

//===========================================================================//

bool RelationAdapter::checkUnique(const string &name, WorkPtr &work)
{
	const result &res = work->prepared("checkUnique")(name).exec();
	return (res[0][0].as<int>() == 0);
}

//===========================================================================//

int RelationAdapter::insertRelation(const Relation &relation, WorkPtr &work)
{
	// Insere informações da relação
	work->prepared("insertRelation")(relation.name)
			(relation.attributes.size())(relation.data.size()).exec();

	// Recupera o ID gerado
	result res = work->prepared("selectLastID").exec();
	return res[0][0].as<int>();
}

//===========================================================================//

void RelationAdapter::insertAttribute(int relationID, uint attrIndex,
		const Attribute &attr, WorkPtr &work)
{
	// Se for numérico
	if (attr.type == NUMERIC)
		work->prepared("insertAttribute")(relationID)(attrIndex)
				(attr.name)(NUMERIC_TYPE)().exec();

	// Se for nominal
	else if (attr.type == NOMINAL)
	{
		work->prepared("insertAttribute")(relationID)(attrIndex)(attr.name)
				(NOMINAL_TYPE)(attr.nominal->size()).exec();

		// Insere os valores nominais
		uint i = 1;
		for (auto it = attr.nominal->begin(); it != attr.nominal->end(); it++)
			work->prepared("insertNominal")(relationID)(attrIndex)(i++)
					(*it).exec();
	}
}

//===========================================================================//

void RelationAdapter::insertInstance(int relationID, uint instIndex,
		const Instance &inst, WorkPtr &work)
{
	// Para cada valor da instância
	for (uint i = 0; i < inst.size(); i++)
	{
		const Value &value = *(inst[i]);

		// Se for numérico
		if (value.type == NUMERIC)
			work->prepared("insertInstance")(relationID)(instIndex)(i + 1)
					(value.number)().exec();

		// Se for nominal
		else if (value.type == NOMINAL)
			work->prepared("insertInstance")(relationID)(instIndex)(i + 1)()
					(value.nominal).exec();

		// Se for vazio
		else if (value.type == EMPTY)
			work->prepared("insertInstance")(relationID)(instIndex)(i + 1)()()
					.exec();
	}
}

//===========================================================================//

}
