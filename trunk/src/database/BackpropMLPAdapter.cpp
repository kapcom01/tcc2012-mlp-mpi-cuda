#include "database/BackpropMLPAdapter.h"
#include "database/DatabaseException.h"

namespace Database
{

//===========================================================================//

void BackpropMLPAdapter::prepareForInsert(connection* conn)
{
	try
	{
		// Verificação de unicidade da relação MLP
		conn->prepare("checkUnique",
				"SELECT COUNT(*) FROM MLP WHERE Name = $1")
				("VARCHAR", prepare::treat_string);

		// Inserção na tabela MLP
		conn->prepare("insertMLP",
				"INSERT INTO MLP (Name, LowerValue, UpperValue, NLayers, "
				"TrainedRelation) VALUES ($1, $2, $3, $4, NULL)")
				("VARCHAR", prepare::treat_string)("NUMERIC")("NUMERIC")
				("INTEGER");

		// Inserção na tabela Layer
		conn->prepare("insertLayer",
				"INSERT INTO Layer (MLPID, LayerIndex, NInputs, NNeurons) "
				"VALUES ($1, $2, $3, $4)")
				("INTEGER")("INTEGER")("INTEGER")("INTEGER");

		// Inserção na tabela Neuron
		conn->prepare("insertNeuron",
				"INSERT INTO Neuron (MLPID, LayerIndex, NeuronIndex, "
				"InputIndex, Weight) VALUES ($1, $2, $3, $4, $5)")
				("INTEGER")("INTEGER")("INTEGER")("INTEGER")("NUMERIC");

		// Seleção do último ID de MLP gerado
		conn->prepare("selectLastID",
				"SELECT currval(pg_get_serial_sequence('MLP', 'mlpid'))");
	}
	catch (pqxx_exception &e)
	{
		throw DatabaseException(e);
	}
}

//===========================================================================//

void BackpropMLPAdapter::insert(BackpropMLP &mlp)
{
	// Cria uma nova conexão com a base de dados
	Connection conn;
	prepareForInsert(conn.get());

	WorkPtr work = conn.getWork();

	try
	{
		// Verifica se o nome da relação é realmente único
		if (!checkUnique(mlp.name, work))
			throw DatabaseException(RELATION_NOT_UNIQUE);

		// Insere as informações do MLP
		mlp.mlpID = insertMLP(mlp, work);

		// Insere as camadas
		for (uint l = 0; l < mlp.layers.size(); l++)
			insertLayer(mlp.mlpID, l + 1, *(mlp.layers[l]), work);

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

bool BackpropMLPAdapter::checkUnique(const string &name, WorkPtr &work)
{
	const result &res = work->prepared("checkUnique")(name).exec();
	return (res[0][0].as<int>() == 0);
}

//===========================================================================//

int BackpropMLPAdapter::insertMLP(const BackpropMLP &mlp, WorkPtr &work)
{
	// Insere informações do MLP
	Range range = mlp.getRange();

	work->prepared("insertMLP")(mlp.name)(range.lower)(range.upper)
			(mlp.layers.size()).exec();

	// Recupera o ID gerado
	const result &res = work->prepared("selectLastID").exec();
	return res[0][0].as<int>();
}

//===========================================================================//

void BackpropMLPAdapter::insertLayer(int mlpID, uint layerIndex,
		const Layer &layer, WorkPtr &work)
{
	// Insere informações da camada
	work->prepared("insertLayer")(mlpID)(layerIndex)(layer.inUnits)
			(layer.outUnits).exec();

	// Insere os neurônios
	for (uint n = 0; n < layer.outUnits; n++)
		insertNeuron(mlpID, layerIndex, n + 1, *(layer.neurons[n]), work);
}

//===========================================================================//

void BackpropMLPAdapter::insertNeuron(int mlpID, uint layerIndex,
		uint neuronIndex, const Neuron &neuron, WorkPtr &work)
{
	// Insere cada um dos pesos
	for (uint i = 0; i <= neuron.inUnits; i++)
		work->prepared("insertNeuron")(mlpID)(layerIndex)(neuronIndex)(i + 1)
				(neuron.weights[i]).exec();
}

//===========================================================================//

void BackpropMLPAdapter::prepareForSelect(connection* conn)
{
	try
	{
		// Seleção na tabela MLP
		conn->prepare("selectMLP",
				"SELECT Name, LowerValue, UpperValue FROM MLP "
				"WHERE MLPID = $1")
				("INTEGER");

		// Seleção na tabela Layer
		conn->prepare("selectLayer",
				"SELECT NInputs, NNeurons FROM Layer WHERE MLPID = $1 "
				"ORDER BY LayerIndex")
				("INTEGER");

		// Seleção na tabela Neuron
		conn->prepare("selectNeuron",
				"SELECT Weight FROM Neuron WHERE MLPID = $1 AND "
				"LayerIndex = $2 AND NeuronIndex = $3 "
				"ORDER BY InputIndex")
				("INTEGER")("INTEGER")("INTEGER");
	}
	catch (pqxx_exception &e)
	{
		throw DatabaseException(e);
	}
}

//===========================================================================//

void BackpropMLPAdapter::select(BackpropMLP &mlp)
{
	// Cria uma nova conexão com a base de dados
	Connection conn;
	prepareForSelect(conn.get());

	WorkPtr work = conn.getWork();

	try
	{
		// Recupera as informações do MLP
		selectMLP(mlp, work);

		// Recupera as camadas
		for (uint l = 0; l < mlp.layers.size(); l++)
			selectLayer(mlp.mlpID, l + 1, *(mlp.layers[l]), work);

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

void BackpropMLPAdapter::selectMLP(BackpropMLP &mlp, WorkPtr &work)
{
	const result &res = work->prepared("selectMLP")(mlp.mlpID).exec();

	mlp.name = res[0]["Name"].as<string>();
	mlp.range.lower = res[0]["LowerValue"].as<double>();
	mlp.range.upper = res[0]["UpperValue"].as<double>();

	const result &layers = work->prepared("selectLayer")(mlp.mlpID).exec();

	// Adiciona as camadas
	for (auto row = layers.begin(); row != layers.end(); row++)
	{
		uint inUnits = row["NInputs"].as<uint>();
		uint outUnits = row["NNeurons"].as<uint>();

		LayerPtr layer(new Layer(inUnits, outUnits));
		mlp.layers.push_back(layer);

		if (row + 1 == layers.end())
		{
			mlp.output = &(layer->funcSignal);
			mlp.error.resize(outUnits);
		}
	}
}

//===========================================================================//

void BackpropMLPAdapter::selectLayer(int mlpID, uint layerIndex, Layer &layer,
		WorkPtr &work)
{
	// Recupera os pesos de cada neurônio
	for (uint n = 0; n < layer.outUnits; n++)
		selectNeuron(mlpID, layerIndex, n + 1, *(layer.neurons[n]), work);
}

//===========================================================================//

void BackpropMLPAdapter::selectNeuron(int mlpID, uint layerIndex,
		uint neuronIndex, Neuron &neuron, WorkPtr &work)
{
	// Seleciona os pesos dos neurônios
	const result &res = work->prepared("selectNeuron")(mlpID)(layerIndex)
			(neuronIndex).exec();

	// Seta os pesos
	uint i = 0;
	for (auto row = res.begin(); row != res.end(); row++)
		neuron.weights[i++] = row[0].as<double>();
}

//===========================================================================//

void BackpropMLPAdapter::prepareForUpdate(connection* conn)
{
	try
	{
		// Adiciona a relação treinada
		conn->prepare("updateRelation",
				"UPDATE MLP SET TrainedRelation = $2 WHERE MLPID = $1")
				("INTEGER")("INTEGER");

		// Atualização na tabela Neuron
		conn->prepare("updateNeuron",
				"UPDATE Neuron SET Weight = $5 WHERE MLPID = $1 "
				"AND LayerIndex = $2 AND NeuronIndex = $3 AND InputIndex = $4")
				("INTEGER")("INTEGER")("INTEGER")("INTEGER")("NUMERIC");
	}
	catch (pqxx_exception &e)
	{
		throw DatabaseException(e);
	}
}

//===========================================================================//

void BackpropMLPAdapter::update(const BackpropMLP &mlp, int relationID)
{
	// Cria uma nova conexão com a base de dados
	Connection conn;
	prepareForUpdate(conn.get());

	WorkPtr work = conn.getWork();

	try
	{
		// Adiciona a relação com qual a rede foi treinada
		updateRelation(mlp.mlpID, relationID, work);

		// Atualiza cada camada
		for (uint l = 0; l < mlp.layers.size(); l++)
			updateLayer(mlp.mlpID, l + 1, *(mlp.layers[l]), work);

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

void BackpropMLPAdapter::updateRelation(int mlpID, int relationID,
		WorkPtr &work)
{
	work->prepared("updateRelation")(mlpID)(relationID).exec();
}

//===========================================================================//

void BackpropMLPAdapter::updateLayer(int mlpID, uint layerIndex,
		const Layer &layer, WorkPtr &work)
{
	// Insere os pesos de cada neurônio
	for (uint n = 0; n < layer.outUnits; n++)
		updateNeuron(mlpID, layerIndex, n + 1, *(layer.neurons[n]), work);
}

//===========================================================================//

void BackpropMLPAdapter::updateNeuron(int mlpID, uint layerIndex,
		uint neuronIndex, const Neuron &neuron, WorkPtr &work)
{
	// Atualiza cada um dos pesos
	for (uint i = 0; i <= neuron.inUnits; i++)
		work->prepared("updateNeuron")(mlpID)(layerIndex)(neuronIndex)(i + 1)
				(neuron.weights[i]).exec();
}

//===========================================================================//

}
