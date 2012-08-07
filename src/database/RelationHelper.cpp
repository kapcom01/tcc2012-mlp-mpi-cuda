#include "database/RelationHelper.h"
#include "database/DatabaseException.h"

namespace Database
{

//===========================================================================//

void RelationHelper::prepare(connection* conn)
{
	try
	{
		// Verificação de unicidade da relação Relation
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

void RelationHelper::insert(const Relation &relation)
{
	// Cria uma nova conexão com a base de dados
	Connection conn;
	prepare(conn.get());

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

bool RelationHelper::checkUnique(const string &name, WorkPtr &work)
{
	result res = work->prepared("checkUnique")(name).exec();
	return (res[0][0].as<int>() == 0);
}

//===========================================================================//

int RelationHelper::insertRelation(const Relation &relation, WorkPtr &work)
{
	// Insere informações da relação
	work->prepared("insertRelation")(relation.name)(
			relation.attributes.size())(relation.data.size()).exec();

	// Recupera o ID gerado
	result res = work->prepared("selectLastID").exec();
	return res[0][0].as<int>();
}

//===========================================================================//

void RelationHelper::insertAttribute(uint relationID, uint attrIndex,
		const Attribute &attr, WorkPtr &work)
{
	// Se for numérico
	if (attr.type == NUMERIC)
		work->prepared("insertAttribute")(relationID)(attrIndex)
				(attr.name)(1)().exec();

	// Se for nominal
	else if (attr.type == NOMINAL)
	{
		work->prepared("insertAttribute")(relationID)(attrIndex)(attr.name)(2)
				(attr.nominal->size()).exec();

		// Insere os valores nominais
		for (const string &str : *(attr.nominal))
			work->prepared("insertNominal")(relationID)(attrIndex)(str).exec();
	}
}

//===========================================================================//

void RelationHelper::insertInstance(uint relationID, uint instIndex,
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
					(*(value.str)).exec();

		// Se for vazio
		else if (value.type == EMPTY)
			work->prepared("insertInstance")(relationID)(instIndex)(i + 1)()()
					.exec();
	}
}

//===========================================================================//

}
