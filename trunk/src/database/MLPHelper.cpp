#include "database/MLPHelper.h"
#include "database/DatabaseException.h"

namespace Database
{

//===========================================================================//

void MLPHelper::prepare(connection* conn)
{
	try
	{
		// Verificação de unicidade da relação MLP
		conn->prepare("checkUnique",
				"SELECT COUNT(*) FROM MLP WHERE Name = $1")
				("VARCHAR", prepare::treat_string);

		// Inserção na tabela MLP
		conn->prepare("insertMLP",
				"INSERT INTO MLP (Name, ActivationType, ProblemType, NLayers, Status) VALUES ($1, $2, $3, $4, 1)")
				("VARCHAR", prepare::treat_string)("SMALLINT")("SMALLINT")
				("INTEGER");

		// Inserção na tabela Layer
		conn->prepare("insertLayer",
				"INSERT INTO Layer (MLPID, LayerIndex, NInputs, NNeurons) VALUES ($1, $2, $3, $4)")
				("INTEGER")("INTEGER")("INTEGER")("INTEGER");

		// Inserção na tabela Neuron
		conn->prepare("insertNeuron",
				"INSERT INTO Neuron (MLPID, LayerIndex, NeuronIndex, InputIndex, Weight) VALUES ($1, $2, $3, $4, $5)")
				("INTEGER")("INTEGER")("INTEGER")("INTEGER")("NUMERIC");

		// Atualização na tabela Neuron
		conn->prepare("updateNeuron",
				"UPDATE Neuron (Weight) SET ($1, $2, $3, $4, $5) WHERE MLPID = $1 AND LayerIndex = $2 AND NeuronIndex = $3 AND InputIndex = $4")
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

void MLPHelper::insert(BackpropMLP &mlp, const string &name)
{
	// Cria uma nova conexão com a base de dados
	Connection conn;
	prepare(conn.get());

	WorkPtr work = conn.getWork();

	try
	{
		// Verifica se o nome da relação é realmente único
		if (!checkUnique(name, work))
			throw DatabaseException(RELATION_NOT_UNIQUE);

		// Insere as informações do MLP
		mlp.mlpID = insertMLP(mlp, name, work);

		// Insere as camadas
		for (uint i = 0; i < mlp.nLayers; i++)
			insertLayer(mlp.mlpID, i + 1, *(mlp.layers[i]), work);

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

void MLPHelper::update(const BackpropMLP &mlp)
{
	// Cria uma nova conexão com a base de dados
	Connection conn;
	prepare(conn.get());

	WorkPtr work = conn.getWork();

	try
	{
		// Para cada camada
		for (uint i = 0; i < mlp.nLayers; i++)
		{
			Layer* layer = mlp.layers[i];

			// Atualização dos pesos para cada neurônio
			for (uint j = 0; j < layer->outUnits; j++)
				updateNeuron(mlp.mlpID, i + 1, j + 1, layer->weights[j],
						layer->inUnits, work);
		}

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

bool MLPHelper::checkUnique(const string &name, WorkPtr &work)
{
	result res = work->prepared("checkUnique")(name).exec();
	return (res[0][0].as<int>() == 0);
}

//===========================================================================//

int MLPHelper::insertMLP(const BackpropMLP &mlp, const string &name,
		WorkPtr &work)
{
	// Insere informações do MLP
	work->prepared("insertMLP")(name)((int) mlp.activationType)
			((int) mlp.problemType)(mlp.nLayers).exec();

	// Recupera o ID gerado
	result res = work->prepared("selectLastID").exec();
	return res[0][0].as<int>();
}

//===========================================================================//

void MLPHelper::insertLayer(int mlpID, uint layerIndex, const Layer &layer,
			WorkPtr &work)
{
	// Insere informações da camada
	work->prepared("insertLayer")(mlpID)(layerIndex)(layer.inUnits)
			(layer.outUnits).exec();

	// Insere os pesos de cada neurônio
	for (uint i = 0; i < layer.outUnits; i++)
		insertNeuron(mlpID, layerIndex, i + 1, layer.weights[i], layer.inUnits,
				work);
}

//===========================================================================//

void MLPHelper::insertNeuron(int mlpID, uint layerIndex, uint neuronIndex,
		const double* weights, uint inUnits, WorkPtr &work)
{
	// Para cada peso
	for (uint i = 0; i <= inUnits; i++)
		work->prepared("insertNeuron")(mlpID)(layerIndex)(neuronIndex)
				(i + 1)(weights[i]);
}

//===========================================================================//

void MLPHelper::updateNeuron(int mlpID, uint layerIndex, uint neuronIndex,
		const double* weights, uint inUnits, WorkPtr &work)
{
	// Para cada peso
	for (uint i = 0; i <= inUnits; i++)
		work->prepared("updateNeuron")(mlpID)(layerIndex)(neuronIndex)
				(i + 1)(weights[i]);
}

//===========================================================================//

}
