#include "database/ExampleSetAdapter.h"
#include "database/DatabaseException.h"

namespace Database
{

//===========================================================================//

void ExampleSetAdapter::prepareForSelect(connection* conn)
{
	try
	{
		// Seleção da relação de treinamento
		conn->prepare("selectTrainedRelation",
				"SELECT TrainedRelation FROM MLP WHERE MLPID = $1")
				("INTEGER");

		// Seleção da quantidade de atributos de uma relação
		conn->prepare("selectNAttr",
				"SELECT NAttributes FROM Relation WHERE RelationID = $1")
				("INTEGER");

		// Seleção do intervalo de valores do MLP
		conn->prepare("selectRange",
				"SELECT LowerValue, UpperValue FROM MLP WHERE MLPID = $1")
				("INTEGER");

		// Seleção das estatísticas
		conn->prepare("selectStatistics",
				"SELECT Type, NominalCard, Minimum, Maximum "
				"FROM Attribute WHERE RelationID = $1 ORDER BY AttrIndex")
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

void ExampleSetAdapter::select(ExampleSet &set)
{
	// Cria uma nova conexão com a base de dados
	Connection conn;
	prepareForSelect(conn.get());

	WorkPtr work = conn.getWork();

	try
	{
		// Seleciona a quantidade de atributos
		int nattr = selectNAttributes(set.relationID, work);

		// Seleciona os dados
		selectData(set, nattr, work);

		// Seleciona as estatísticas
		selectStatistics(set, work);

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

int ExampleSetAdapter::selectNAttributes(int relationID, WorkPtr &work)
{
	const result &res = work->prepared("selectNAttr")(relationID).exec();
	return res[0][0].as<int>();
}

//===========================================================================//

void ExampleSetAdapter::selectData(ExampleSet &set, int nattr, WorkPtr &work)
{
	const result &res = work->prepared("selectData")(set.relationID).exec();

	// Para cada tupla do resultado
	for (auto row = res.begin(); row != res.end(); row++)
	{
		int attrIndex = row["AttrIndex"].as<int>();

		// Adiciona uma nova instância
		if (attrIndex == 1)
		{
			set.input.push_back(vector<double>());
			set.target.push_back(vector<double>());
			set.output.push_back(vector<double>());
		}

		// Se for numérico
		if (row["Type"].as<int>() == NUMERIC_TYPE)
		{
			double value = row["NumericValue"].as<double>();
			addValue(set, value, attrIndex == nattr && set.type != TEST);
		}

		// Se for nominal
		else
		{
			double value = row["NominalValue"].as<int>();
			int card = row["NominalCard"].as<int>();
			addValue(set, value, card, attrIndex == nattr && set.type != TEST);
		}
	}
}

//===========================================================================//

int ExampleSetAdapter::selectTrainedRelation(ExampleSet &set,
		WorkPtr &work)
{
	// Se for conjunto de treinamento, retorna próprio ID
	if (set.type == TRAINING)
		return set.relationID;
	// Se for outro tipo, retorna a relação de treinamento do MLP
	else
	{
		const result &res = work->prepared("selectTrainedRelation")
				(set.mlpID).exec();
		return res[0][0].as<int>();
	}
}

//===========================================================================//

Range ExampleSetAdapter::selectRange(int mlpID, WorkPtr &work)
{
	const result &res = work->prepared("selectRange")(mlpID).exec();
	return {res[0][0].as<double>(), res[0][1].as<double>()};
}

//===========================================================================//

void ExampleSetAdapter::selectStatistics(ExampleSet &set, WorkPtr &work)
{
	// Seleciona a relação de treinamento
	int trained = selectTrainedRelation(set, work);

	// Seleciona o intervalo de valores do MLP
	Range range = selectRange(set.mlpID, work);

	const result &res = work->prepared("selectStatistics")(trained).exec();

	// Para cada tupla do resultado
	for (auto row = res.begin(); row != res.end(); row++)
	{
		// Se for numérico
		if (row["Type"].as<int>() == NUMERIC_TYPE)
		{
			double min = row["Minimum"].as<double>();
			double max = row["Maximum"].as<double>();

			set.stat.push_back({{min, max}, {-1, 1}});
		}

		// Se for nominal
		else
		{
			uint card = row["NominalCard"].as<uint>();

			for (uint i = 0; i < card; i++)
				set.stat.push_back({{0, 1}, {range.lower, range.upper}});
		}
	}
}

//===========================================================================//

void ExampleSetAdapter::addValue(ExampleSet &set, double value, bool isTarget)
{
	// Se for saída
	if (isTarget)
	{
		set.target.back().push_back(value);
		set.output.back().push_back(0);
	}
	// Se for entrada
	else
		set.input.back().push_back(value);
}

//===========================================================================//

void ExampleSetAdapter::addValue(ExampleSet &set, int value, uint card,
		bool isTarget)
{
	// Adiciona uma variável para cada possível valor
	for (uint i = 0; i < card; i++)
		if (i + 1 == value)
			addValue(set, 1, isTarget);
		else
			addValue(set, 0, isTarget);
}

//===========================================================================//

void ExampleSetAdapter::prepareForInsert(connection* conn)
{
	try
	{
		// Inserção na tabela Operation
		conn->prepare("insertOperation",
				"INSERT INTO Operation (Type, MLPID, RelationID, "
				"LearningRate, Tolerance, MaxEpochs, Error, Time) VALUES "
				"($1, $2, $3, $4, $5, $6, $7, $8)")
				("SMALLINT")("INTEGER")("INTEGER")("NUMERIC")("NUMERIC")
				("INTEGER")("NUMERIC")("NUMERIC");

		// Seleção do tipo do atributo de saída
		conn->prepare("selectType",
				"SELECT A.Type FROM Attribute A, Relation R WHERE "
				"A.AttrIndex = R.NAttributes AND A.RelationID = $1 "
				"AND R.RelationID = $1")
				("INTEGER");

		// Inserção na tabela Results
		conn->prepare("insertResults",
				"INSERT INTO Result (OperationID, InstanceIndex, "
				"NumericValue, NominalValue) VALUES ($1, $2, $3, $4)")
				("INTEGER")("INTEGER")("NUMERIC")("SMALLINT");

		// Seleção do último ID de Operation gerado
		conn->prepare("selectLastID",
				"SELECT currval(pg_get_serial_sequence('Operation', "
				"'operationid'))");
	}
	catch (pqxx_exception &e)
	{
		throw DatabaseException(e);
	}
}

//===========================================================================//

void ExampleSetAdapter::insert(const ExampleSet &set)
{
	// Cria uma nova conexão com a base de dados
	Connection conn;
	prepareForInsert(conn.get());

	WorkPtr work = conn.getWork();

	try
	{
		// Insere as informações da operação
		int opID = insertOperation(set, work);

		// Insere os resultados
		insertResults(opID, set, work);

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

int ExampleSetAdapter::insertOperation(const ExampleSet &set, WorkPtr &work)
{
	work->prepared("insertOperation")((int) set.type)(set.mlpID)
			(set.relationID)(set.learning)(set.tolerance)(set.maxEpochs)
			(set.error)(set.time).exec();

	// Recupera o ID gerado
	const result &res = work->prepared("selectLastID").exec();
	return res[0][0].as<int>();
}

//===========================================================================//

bool ExampleSetAdapter::selectType(const ExampleSet &set, WorkPtr &work)
{
	const result &res = work->prepared("selectType")(set.relationID).exec();

	// Verifica o tipo
	return (res[0]["Type"].as<int>() == 1);
}

//===========================================================================//

void ExampleSetAdapter::insertResults(int opID, const ExampleSet &set,
		WorkPtr &work)
{
	// Para cada instância
	for (uint i = 0; i < set.size(); i++)
	{
		// Se for do tipo numérico
		if (selectType(set, work))
		{
			double value = set.output[i][0];
			work->prepared("insertResults")(opID)(i + 1)(value)().exec();
		}

		// Se for do tipo nominal
		else
		{
			auto it = max_element(set.output[i].begin(), set.output[i].end());
			int value = it - set.output[i].begin() + 1;
			work->prepared("insertResults")(opID)(i + 1)()(value).exec();
		}
	}
}

//===========================================================================//

}
