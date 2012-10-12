#include "database/ExampleSetAdapter.h"

namespace ParallelMLP
{

//===========================================================================//

void ExampleSetAdapter::prepareForSelect(connection &conn)
{
	try
	{
		// Seleção da relação de treinamento
		conn.prepare("selectTrainedRelation",
				"SELECT TrainedRelation FROM MLP WHERE MLPID = $1")
				("INTEGER");

		// Seleção do intervalo de valores do MLP
		conn.prepare("selectRange",
				"SELECT LowerValue, UpperValue FROM MLP WHERE MLPID = $1")
				("INTEGER");

		// Seleção das estatísticas
		conn.prepare("selectStatistics",
				"SELECT AttrIndex, Type, NominalCard, Minimum, Maximum "
				"FROM Attribute WHERE RelationID = $1 ORDER BY AttrIndex")
				("INTEGER");

		// Selação da quantidade de instâncias
		conn.prepare("selectSize",
				"SELECT NInstances, NAttributes FROM Relation "
				"WHERE RelationID = $1")
				("INTEGER");

		// Selação dos dados da relação
		conn.prepare("selectData",
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
		throw ParallelMLPException(e.base());
	}
}

//===========================================================================//

void ExampleSetAdapter::select(ExampleSet &set)
{
	// Cria uma nova conexão com a base de dados
	prepareForSelect(Connection::get());
	work work(Connection::get());

	try
	{
		// Seleciona a quantidade de instâncias e atributos
		Size size = selectSize(set.getID(), work);

		// Seleciona os dados
		selectData(set, size, work);

		// Seleciona as estatísticas
		selectStatistics(set, size, work);

		// Salva as alterações
		work.commit();
	}
	catch (pqxx_exception &e)
	{
		// Desfaz as alterações
		work.abort();
		throw ParallelMLPException(e.base());
	}
}

//===========================================================================//

Size ExampleSetAdapter::selectSize(int relationID, work &work)
{
	const result &res = work.prepared("selectSize")(relationID).exec();
	uint nInst = res[0]["NInstances"].as<uint>();
	uint nAttr = res[0]["NAttributes"].as<uint>();
	return { nInst, nAttr };
}

//===========================================================================//

void ExampleSetAdapter::selectData(ExampleSet &set, Size &size, work &work)
{
	// Seta a quantidade de instâncias
	set.setSize(size.nInst);

	// Para cada instância
	for (uint k = 0; k < size.nInst; k++)
	{
		const result &res = work.prepared("selectData")(set.getID())(k + 1)
				.exec();

		for (auto row = res.begin(); row != res.end(); row++)
		{
			uint attrIndex = row["AttrIndex"].as<uint>();
			bool isTarget = attrIndex == size.nAttr && !set.isTest();

			// Se for numérico
			if (row["Type"].as<int>() == NUMERIC_TYPE)
			{
				float value = row["NumericValue"].as<float>();
				set.addValue(value, isTarget);
			}

			// Se for nominal
			else
			{
				float value = row["NominalValue"].as<int>();
				int card = row["NominalCard"].as<int>();
				set.addValue(value, card, isTarget);
			}
		}
	}

	set.done();
}

//===========================================================================//

int ExampleSetAdapter::selectTrainedRelation(ExampleSet &set,
		work &work)
{
	// Se for conjunto de treinamento, retorna próprio ID
	if (set.isTraining())
		return set.getID();
	// Se for outro tipo, retorna a relação de treinamento do MLP
	else
	{
		const result &res = work.prepared("selectTrainedRelation")
				(set.getMLPID()).exec();
		return res[0][0].as<int>();
	}
}

//===========================================================================//

Range ExampleSetAdapter::selectRange(int mlpID, work &work)
{
	const result &res = work.prepared("selectRange")(mlpID).exec();
	return {res[0][0].as<float>(), res[0][1].as<float>()};
}

//===========================================================================//

void ExampleSetAdapter::selectStatistics(ExampleSet &set, Size &size,
		work &work)
{
	// Seleciona a relação de treinamento
	int trained = selectTrainedRelation(set, work);

	// Seleciona o intervalo de valores do MLP
	Range range = selectRange(set.getMLPID(), work);

	const result &res = work.prepared("selectStatistics")(trained).exec();

	// Para cada tupla do resultado
	for (auto row = res.begin(); row != res.end(); row++)
	{
		uint attrIndex = row["AttrIndex"].as<uint>();
		bool isTarget = attrIndex == size.nAttr && !set.isTest();

		// Se for numérico
		if (row["Type"].as<int>() == NUMERIC_TYPE)
		{
			float min = row["Minimum"].as<float>();
			float max = row["Maximum"].as<float>();
			set.addStat(min, max, -1, 1, isTarget);
		}

		// Se for nominal
		else
		{
			uint card = row["NominalCard"].as<uint>();
			set.addStat(range.lower, range.upper, card, isTarget);
		}
	}
}



//===========================================================================//

void ExampleSetAdapter::prepareForInsert(connection &conn)
{
	try
	{
		// Inserção na tabela Operation
		conn.prepare("insertOperation",
				"INSERT INTO Operation (Type, MLPID, RelationID, "
				"LearningRate, Tolerance, MaxEpochs, Error, Epochs, Time) "
				"VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)")
				("SMALLINT")("INTEGER")("INTEGER")("NUMERIC")("NUMERIC")
				("INTEGER")("NUMERIC")("INTEGER")("NUMERIC");

		// Seleção do tipo do atributo de saída
		conn.prepare("selectType",
				"SELECT A.Type FROM Attribute A, Relation R WHERE "
				"A.AttrIndex = R.NAttributes AND A.RelationID = $1 "
				"AND R.RelationID = $1")
				("INTEGER");

		// Inserção na tabela Results
		conn.prepare("insertResults",
				"INSERT INTO Result (OperationID, InstanceIndex, "
				"NumericValue, NominalValue) VALUES ($1, $2, $3, $4)")
				("INTEGER")("INTEGER")("NUMERIC")("SMALLINT");

		// Seleção do último ID de Operation gerado
		conn.prepare("selectLastID",
				"SELECT currval(pg_get_serial_sequence('Operation', "
				"'operationid'))");
	}
	catch (pqxx_exception &e)
	{
		throw ParallelMLPException(e.base());
	}
}

//===========================================================================//

void ExampleSetAdapter::insert(const ExampleSet &set)
{
	// Cria uma nova conexão com a base de dados
	prepareForInsert(Connection::get());
	work work(Connection::get());

	try
	{
		// Insere as informações da operação
		int opID = insertOperation(set, work);

		// Insere os resultados
		insertResults(opID, set, work);

		// Salva as alterações
		work.commit();
	}
	catch (pqxx_exception &e)
	{
		// Desfaz as alterações
		work.abort();
		throw ParallelMLPException(e.base());
	}
}

//===========================================================================//

int ExampleSetAdapter::insertOperation(const ExampleSet &set, work &work)
{
	work.prepared("insertOperation")(set.getType())(set.getMLPID())
			(set.getID())(set.getLearning())(set.getTolerance())
			(set.getMaxEpochs())(set.getError())(set.getEpochs())
			(set.getTime()).exec();

	// Recupera o ID gerado
	const result &res = work.prepared("selectLastID").exec();
	return res[0][0].as<int>();
}

//===========================================================================//

bool ExampleSetAdapter::selectType(const ExampleSet &set, work &work)
{
	const result &res = work.prepared("selectType")(set.getID()).exec();

	// Verifica o tipo
	return (res[0]["Type"].as<int>() == 1);
}

//===========================================================================//

void ExampleSetAdapter::insertResults(int opID, const ExampleSet &set,
		work &work)
{
	// Para cada instância
	for (uint i = 0; i < set.getSize(); i++)
	{
		// Se for do tipo numérico
		if (selectType(set, work))
		{
			float value = set.getNumericOutput(i);
			work.prepared("insertResults")(opID)(i + 1)(value)().exec();
		}

		// Se for do tipo nominal
		else
		{
			int value = set.getNominalOutput(i);
			work.prepared("insertResults")(opID)(i + 1)()(value).exec();
		}
	}
}

//===========================================================================//

}
