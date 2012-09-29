#include "database/MLPAdapter.h"

namespace ParallelMLP
{

//===========================================================================//

void MLPAdapter::prepareForInsert(connection &conn)
{
	try
	{
		// Verificação de unicidade da relação MLP
		conn.prepare("checkUnique",
				"SELECT COUNT(*) FROM MLP WHERE Name = $1")
				("VARCHAR", prepare::treat_string);

		// Inserção na tabela MLP
		conn.prepare("insertMLP",
				"INSERT INTO MLP (Name, LowerValue, UpperValue, NLayers, "
				"TrainedRelation) VALUES ($1, $2, $3, $4, NULL)")
				("VARCHAR", prepare::treat_string)("NUMERIC")("NUMERIC")
				("INTEGER");

		// Inserção na tabela Layer
		conn.prepare("insertLayer",
				"INSERT INTO Layer (MLPID, LayerIndex, NInputs, NNeurons) "
				"VALUES ($1, $2, $3, $4)")
				("INTEGER")("INTEGER")("INTEGER")("INTEGER");

		// Inserção na tabela Neuron
		conn.prepare("insertNeuron",
				"INSERT INTO Neuron (MLPID, LayerIndex, NeuronIndex, "
				"InputIndex, Weight) VALUES ($1, $2, $3, $4, $5)")
				("INTEGER")("INTEGER")("INTEGER")("INTEGER")("NUMERIC");

		// Seleção do último ID de MLP gerado
		conn.prepare("selectLastID",
				"SELECT currval(pg_get_serial_sequence('MLP', 'mlpid'))");
	}
	catch (pqxx_exception &e)
	{
		throw ParallelMLPException(e.base());
	}
}

//===========================================================================//

void MLPAdapter::insert(MLP &mlp)
{
	// Cria uma nova conexão com a base de dados
	prepareForInsert(Connection::get());
	work work(Connection::get());

	try
	{
		// Verifica se o nome da relação é realmente único
		if (!checkUnique(mlp.getName(), work))
			throw ParallelMLPException(RELATION_NOT_UNIQUE);

		// Insere as informações do MLP
		mlp.setID(insertMLP(mlp, work));

		// Insere as camadas
		for (uint l = 0; l < mlp.getNLayers(); l++)
			insertLayer(mlp.getID(), l + 1, mlp.getLayer(l), work);

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

bool MLPAdapter::checkUnique(const string &name, work &work)
{
	const result &res = work.prepared("checkUnique")(name).exec();
	return (res[0][0].as<int>() == 0);
}

//===========================================================================//

int MLPAdapter::insertMLP(const MLP &mlp, work &work)
{
	// Insere informações do MLP
	Range range = mlp.getRange();

	work.prepared("insertMLP")(mlp.getName())(range.lower)(range.upper)
			(mlp.getNLayers()).exec();

	// Recupera o ID gerado
	const result &res = work.prepared("selectLastID").exec();
	return res[0][0].as<int>();
}

//===========================================================================//

void MLPAdapter::insertLayer(int mlpID, uint layerIndex,
		const Layer &layer, work &work)
{
	// Insere informações da camada
	work.prepared("insertLayer")(mlpID)(layerIndex)(layer.getInUnits())
			(layer.getOutUnits()).exec();

	// Insere os neurônios
	for (uint n = 0; n < layer.getOutUnits(); n++)
		for (uint i = 0; i <= layer.getInUnits(); i++)
			work.prepared("insertNeuron")(mlpID)(layerIndex)(n + 1)(i + 1)
					(layer.getWeight(n, i)).exec();
}

//===========================================================================//

void MLPAdapter::prepareForSelect(connection &conn)
{
	try
	{
		// Seleção na tabela MLP
		conn.prepare("selectMLP",
				"SELECT Name, LowerValue, UpperValue FROM MLP "
				"WHERE MLPID = $1")
				("INTEGER");

		// Seleção na tabela Layer
		conn.prepare("selectLayer",
				"SELECT NInputs, NNeurons FROM Layer WHERE MLPID = $1 "
				"ORDER BY LayerIndex")
				("INTEGER");

		// Seleção na tabela Neuron
		conn.prepare("selectNeuron",
				"SELECT Weight FROM Neuron WHERE MLPID = $1 AND "
				"LayerIndex = $2 AND NeuronIndex = $3 "
				"AND InputIndex = $4")
				("INTEGER")("INTEGER")("INTEGER")("INTEGER");
	}
	catch (pqxx_exception &e)
	{
		throw ParallelMLPException(e.base());
	}
}

//===========================================================================//

void MLPAdapter::select(MLP &mlp)
{
	// Cria uma nova conexão com a base de dados
	prepareForSelect(Connection::get());
	work work(Connection::get());

	try
	{
		// Recupera as informações do MLP
		selectMLP(mlp, work);

		// Recupera as camadas
		for (uint l = 0; l < mlp.getNLayers(); l++)
			selectLayer(mlp.getID(), l + 1, mlp.getLayer(l), work);

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

void MLPAdapter::selectMLP(MLP &mlp, work &work)
{
	const result &res = work.prepared("selectMLP")(mlp.getID()).exec();

	mlp.setName(res[0]["Name"].as<string>());

	float lower = res[0]["LowerValue"].as<float>();
	float upper = res[0]["UpperValue"].as<float>();
	mlp.setRange({ lower, upper });

	const result &layers = work.prepared("selectLayer")(mlp.getID()).exec();

	// Adiciona as camadas
	for (auto row = layers.begin(); row != layers.end(); row++)
	{
		uint inUnits = row["NInputs"].as<uint>();
		uint outUnits = row["NNeurons"].as<uint>();

		mlp.addLayer(inUnits, outUnits);
	}

	// Seta a saída do MLP
	mlp.setOutput();
}

//===========================================================================//

void MLPAdapter::selectLayer(int mlpID, uint layerIndex, Layer &layer,
		work &work)
{
	// Recupera os pesos de cada neurônio
	for (uint n = 0; n < layer.getOutUnits(); n++)
		for (uint i = 0; i <= layer.getInUnits(); i++)
		{
			// Seleciona o peso
			const result &res = work.prepared("selectNeuron")(mlpID)
					(layerIndex)(n + 1)(i + 1).exec();

			// Seta o peso
			layer.setWeight(n, i, res[0][0].as<float>());
		}
}

//===========================================================================//

void MLPAdapter::prepareForUpdate(connection &conn)
{
	try
	{
		// Adiciona a relação treinada
		conn.prepare("updateRelation",
				"UPDATE MLP SET TrainedRelation = $2 WHERE MLPID = $1")
				("INTEGER")("INTEGER");

		// Atualização na tabela Neuron
		conn.prepare("updateNeuron",
				"UPDATE Neuron SET Weight = $5 WHERE MLPID = $1 "
				"AND LayerIndex = $2 AND NeuronIndex = $3 AND InputIndex = $4")
				("INTEGER")("INTEGER")("INTEGER")("INTEGER")("NUMERIC");
	}
	catch (pqxx_exception &e)
	{
		throw ParallelMLPException(e.base());
	}
}

//===========================================================================//

void MLPAdapter::update(const MLP &mlp, int relationID)
{
	// Cria uma nova conexão com a base de dados
	prepareForUpdate(Connection::get());
	work work(Connection::get());

	try
	{
		// Adiciona a relação com qual a rede foi treinada
		updateRelation(mlp.getID(), relationID, work);

		// Atualiza cada camada
		for (uint l = 0; l < mlp.getNLayers(); l++)
			updateLayer(mlp.getID(), l + 1, mlp.getLayer(l), work);

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

void MLPAdapter::updateRelation(int mlpID, int relationID,
		work &work)
{
	work.prepared("updateRelation")(mlpID)(relationID).exec();
}

//===========================================================================//

void MLPAdapter::updateLayer(int mlpID, uint layerIndex,
		const Layer &layer, work &work)
{
	// Insere os pesos de cada neurônio
	for (uint n = 0; n < layer.getOutUnits(); n++)
		for (uint i = 0; i <= layer.getInUnits(); i++)
			work.prepared("updateNeuron")(mlpID)(layerIndex)(n + 1)(i + 1)
					(layer.getWeight(n, i)).exec();
}

//===========================================================================//

}
