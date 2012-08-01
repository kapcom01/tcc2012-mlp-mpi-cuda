#include "database/Inserter.h"
#include "database/DatabaseException.h"

namespace Database
{

//===========================================================================//

void Inserter::insert(const DataSet &dataset)
{
	// Cria uma nova conexão com a base de dados
	Connection conn;
	WorkPtr work = conn.getWork();

	try
	{
		// Verifica se o nome da relação é realmente único
		if (!checkUnique(dataset.relation, work))
			throw DatabaseException(RELATION_NOT_UNIQUE);

		// Insere a relação
		int relationID = insertRelation(dataset, work);

		// Insere os atributos
		for (uint i = 0; i < dataset.attributes.size(); i++)
			insertAttribute(relationID, i + 1, *(dataset.attributes[i]), work);

		// Insere os dados
		for (uint i = 0; i < dataset.data.size(); i++)
			insertInstance(relationID, i + 1, *(dataset.data[i]), work);

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

bool Inserter::checkUnique(const string &name, WorkPtr &work)
{
	result res = work->prepared("checkUnique")(name).exec();
	return (res[0][0].as<int>() == 0);
}

//===========================================================================//

int Inserter::insertRelation(const DataSet &dataset, WorkPtr &work)
{
	// Insere informações da relação
	work->prepared("insertRelation")(dataset.relation)(
			dataset.attributes.size())(dataset.data.size()).exec();

	// Recupera o ID gerado
	result res = work->prepared("selectLastID").exec();
	return res[0][0].as<int>();
}

//===========================================================================//

void Inserter::insertAttribute(uint relationID, uint attrIndex,
		const Attribute &attr, WorkPtr &work)
{
	// Se for numérico
	if (attr.type == NUMERIC)
		work->prepared("insertAttribute")(relationID)(attrIndex)(attr.name)(1)().exec();

	// Se for nominal
	else if (attr.type == NOMINAL)
	{
		work->prepared("insertAttribute")(relationID)(attrIndex)(attr.name)(2)(
				attr.nominal->size()).exec();

		// Insere os valores nominais
		for (const string &str : *(attr.nominal))
			work->prepared("insertNominal")(relationID)(attrIndex)(str).exec();
	}
}

//===========================================================================//

void Inserter::insertInstance(uint relationID, uint instIndex,
		const Instance &inst, WorkPtr &work)
{
	// Para cada valor da instância
	for (uint i = 0; i < inst.size(); i++)
	{
		const Value &value = *(inst[i]);

		// Se for numérico
		if (value.type == NUMERIC)
			work->prepared("insertInstance")(relationID)(instIndex)(i + 1)(
					value.number)().exec();

		// Se for nominal
		else if (value.type == NOMINAL)
			work->prepared("insertInstance")(relationID)(instIndex)(i + 1)()(
					*(value.str)).exec();

		// Se for vazio
		else if (value.type == EMPTY)
			work->prepared("insertInstance")(relationID)(instIndex)(i + 1)()().exec();
	}
}

//===========================================================================//

}
