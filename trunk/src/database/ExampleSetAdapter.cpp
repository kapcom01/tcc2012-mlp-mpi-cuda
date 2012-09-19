#include "database/ExampleSetAdapter.h"
#include "database/DatabaseException.h"

namespace ParallelMLP
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

		// Seleção do intervalo de valores do MLP
		conn->prepare("selectRange",
				"SELECT LowerValue, UpperValue FROM MLP WHERE MLPID = $1")
				("INTEGER");

		// Seleção das estatísticas
		conn->prepare("selectStatistics",
				"SELECT AttrIndex, Type, NominalCard, Minimum, Maximum "
				"FROM Attribute WHERE RelationID = $1 ORDER BY AttrIndex")
				("INTEGER");

		// Selação da quantidade de instâncias
		conn->prepare("selectSize",
				"SELECT NInstances, NAttributes FROM Relation "
				"WHERE RelationID = $1")
				("INTEGER");

		// Selação dos dados da relação
		conn->prepare("selectData",
				"SELECT D.AttrIndex, A.Type, A.NominalCard, D.NumericValue, "
				"D.NominalValue FROM Attribute A, Data D "
				"WHERE A.AttrIndex = D.AttrIndex AND "
				"A.RelationID = D.RelationID AND A.RelationID = $1 "
				"AND D.InstanceIndex = $2 "
				"ORDER BY D.AttrIndex")
				("INTEGER")("INTEGER");
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
		// Seleciona a quantidade de instâncias e atributos
		Size size = selectSize(set.relationID, work);

		// Seleciona os dados
		selectData(set, size, work);

		// Seleciona as estatísticas
		selectStatistics(set, size, work);

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

Size ExampleSetAdapter::selectSize(int relationID, WorkPtr &work)
{
	const result &res = work->prepared("selectSize")(relationID).exec();
	uint nInst = res[0]["NInstances"].as<uint>();
	uint nAttr = res[0]["NAttributes"].as<uint>();
	return { nInst, nAttr };
}

//===========================================================================//

void ExampleSetAdapter::selectData(ExampleSet &set, Size &size, WorkPtr &work)
{
	set.size = size.nInst;

	// Para cada instância
	for (uint k = 0; k < size.nInst; k++)
	{
		const result &res = work->prepared("selectData")(set.relationID)(k + 1)
				.exec();

		for (auto row = res.begin(); row != res.end(); row++)
		{
			uint attrIndex = row["AttrIndex"].as<uint>();

			// Se for numérico
			if (row["Type"].as<int>() == NUMERIC_TYPE)
			{
				float value = row["NumericValue"].as<float>();
				addValue(set, value,
						attrIndex == size.nAttr && set.type != TEST);
			}

			// Se for nominal
			else
			{
				float value = row["NominalValue"].as<int>();
				int card = row["NominalCard"].as<int>();
				addValue(set, value, card,
						attrIndex == size.nAttr && set.type != TEST);
			}
		}

		// Seta a quantidade de variáveis de entrada e saída
		if (k == 0)
		{
			set.inVars = set.input.size();
			set.outVars = set.target.size();
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
	return {res[0][0].as<float>(), res[0][1].as<float>()};
}

//===========================================================================//

void ExampleSetAdapter::selectStatistics(ExampleSet &set, Size &size,
		WorkPtr &work)
{
	// Seleciona a relação de treinamento
	int trained = selectTrainedRelation(set, work);

	// Seleciona o intervalo de valores do MLP
	Range range = selectRange(set.mlpID, work);

	const result &res = work->prepared("selectStatistics")(trained).exec();

	// Para cada tupla do resultado
	for (auto row = res.begin(); row != res.end(); row++)
	{
		uint attrIndex = row["AttrIndex"].as<uint>();

		// Se for numérico
		if (row["Type"].as<int>() == NUMERIC_TYPE)
		{
			float min = row["Minimum"].as<float>();
			float max = row["Maximum"].as<float>();

			addStat(set, min, max, -1, 1,
					attrIndex == size.nAttr && set.type != TEST);
		}

		// Se for nominal
		else
		{
			uint card = row["NominalCard"].as<uint>();

			addStat(set, range.lower, range.upper, card,
					attrIndex == size.nAttr && set.type != TEST);
		}
	}
}

//===========================================================================//

void ExampleSetAdapter::addValue(ExampleSet &set, float value, bool isTarget)
{
	// Se for saída
	if (isTarget)
	{
		set.target.push_back(value);
		set.output.push_back(0);
	}
	// Se for entrada
	else
		set.input.push_back(value);
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

void ExampleSetAdapter::addStat(ExampleSet &set, float min, float max,
		float lower, float upper, bool isTarget)
{
	// Se for saída
	if (isTarget)
		set.outStat.push_back({ {min, max}, {lower, upper} });
	// Se for entrada
	else
		set.inStat.push_back({ {min, max}, {lower, upper} });
}

//===========================================================================//

void ExampleSetAdapter::addStat(ExampleSet &set, float lower, float upper,
		uint card, bool isTarget)
{
	// Adiciona uma variável para cada possível valor
	for (uint i = 0; i < card; i++)
		addStat(set, 0, 1, lower, upper, isTarget);
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
	for (uint i = 0; i < set.size; i++)
	{
		// Se for do tipo numérico
		if (selectType(set, work))
		{
			float value = set.output[i * set.outVars];
			work->prepared("insertResults")(opID)(i + 1)(value)().exec();
		}

		// Se for do tipo nominal
		else
		{
			uint ind = indexOfMax(&(set.output[i * set.inVars]), set.inVars);
			int value = ind % set.inVars + 1;
			work->prepared("insertResults")(opID)(i + 1)()(value).exec();
		}
	}
}

//===========================================================================//

uint ExampleSetAdapter::indexOfMax(const float* vec, uint size)
{
	float max = vec[0];
	uint ind = 0;

	// Percorre o vetor para encontrar o maior elemento
	for (uint i = 1; i < size; i++)
		if (vec[i] > max)
		{
			max = vec[i];
			ind = i;
		}

	return ind;
}

//===========================================================================//

}
