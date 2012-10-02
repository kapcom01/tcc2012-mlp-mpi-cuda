#include "mlp/common/MLP.h"
#include "exception/ParallelMLPException.h"

namespace ParallelMLP
{

//===========================================================================//

MLP::MLP(int mlpID)
{
	this->mlpID = mlpID;
	this->totalError = 0;
	this->epoch = 0;
}

//===========================================================================//

MLP::MLP(string name, vector<uint> &units)
{
	this->name = name;
	this->mlpID = -1;
	this->totalError = 0;
	this->epoch = 0;

	// Intervalo da função de ativação hiperbólica
	range.lower = -1, range.upper = 1;
}

//===========================================================================//

MLP::~MLP()
{
	for (uint i = 0; i < layers.size(); i++)
		delete layers[i];
}

//===========================================================================//

void MLP::randomize()
{
	for (uint i = 0; i < layers.size(); i++)
		layers[i]->randomize();
}

//===========================================================================//

void MLP::initOperation(ExampleSet &set)
{
	// Verifica a quantidade de entradas e saídas
	if (set.getInVars() != layers.front()->getInUnits())
		throw ParallelMLPException(INVALID_INPUT_VARS);
	else if (set.getOutVars() != layers.back()->getOutUnits())
		throw ParallelMLPException(INVALID_OUTPUT_VARS);

	// Reseta o cronômetro, o erro total e a época
	chrono.reset();
	totalError = 0;
	epoch = 0;

	// Normaliza o conjunto de dados
	set.normalize();

	// Se for treinamento, randomiza os pesos e inicializa os índices
	if (set.isTraining())
	{
		randomize();
		initIndexes(set.getSize());
	}

	// Avisa as camadas para se prepararem para a operação
	for (uint i = 0; i < layers.size(); i++)
		layers[i]->initOperation();
}

//===========================================================================//

void MLP::endOperation(ExampleSet &set)
{
	// Avisa as camadas para se finalizarem a operação
	for (uint i = 0; i < layers.size(); i++)
		layers[i]->endOperation();

	// Desnormaliza o conjunto de dados
	set.unnormalize();

	// Seta o erro e o tempo de execução da operação
	set.setError(totalError);
	set.setTime(chrono.getMiliseconds());
	set.setEpochs(epoch);

	cout << "totalError: " << totalError << endl;
}

//===========================================================================//

void MLP::train(ExampleSet &training)
{
	// Inicializa a operação
	initOperation(training);

	// Épocas
	for (; epoch < training.getMaxEpochs(); epoch++)
	{
		cout << "Epoch " << epoch << endl;

		shuffleIndexes();
		totalError = 0;

		// Para cada entrada
		for (uint i = 0; i < training.getSize(); i++)
		{
			uint r = indexes[i];

			// Realiza o feedforward e salva os valores no conjunto
			feedforward(training.getInput(r));
			training.setOutput(r, output);

			// Calcula o erro e realiza o feedback
			calculateError(training.getTarget(r));
			feedback(training.getLearning());
		}

		totalError /= training.getSize() * training.getOutVars();

		// Condição de parada: erro menor do que um valor tolerado
		if (totalError < training.getTolerance())
			break;
	}

	// Finaliza a operação
	endOperation(training);
}

//===========================================================================//

void MLP::validate(ExampleSet &validation)
{
	// Inicializa a operação
	initOperation(validation);

	// Para cada entrada
	for (uint i = 0; i < validation.getSize(); i++)
	{
		// Realiza o feedforward e salva os valores no conjunto
		feedforward(validation.getInput(i));
		validation.setOutput(i, output);

		// Calcula o erro
		calculateError(validation.getTarget(i));
	}

	// Finaliza a operação
	endOperation(validation);
}

//===========================================================================//

void MLP::test(ExampleSet &test)
{
	// Inicializa a operação
	initOperation(test);

	// Para cada entrada
	for (uint i = 0; i < test.getSize(); i++)
	{
		// Realiza o feedforward e salva os valores no conjunto
		feedforward(test.getInput(i));
		test.setOutput(i, output);
	}

	// Finaliza a operação
	endOperation(test);
}

//===========================================================================//

void MLP::feedforward(const vec_float input)
{
	// Propaga a entrada para a primeira camada escondida
	layers.front()->feedforward(input);

	// Propaga a saída da primeira camada para o restante das camadas
	for (uint i = 1; i < layers.size(); i++)
		layers[i]->feedforward(layers[i - 1]->getFuncSignal());
}

//===========================================================================//

void MLP::feedback(float learning)
{
	// Propaga os erros na camada de saída
	layers.back()->feedback(rawError, learning);

	// Propaga o sinal de erro para o restante das camadas
	for (int i = layers.size() - 2; i >= 0; i--)
		layers[i]->feedback(layers[i + 1]->getErrorSignal(), learning);
}

//===========================================================================//

void MLP::initIndexes(uint size)
{
	indexes.resize(size);
	for (uint i = 0; i < indexes.size(); i++)
		indexes[i] = i;
}

//===========================================================================//

void MLP::shuffleIndexes()
{
	for (uint i = indexes.size() - 1; i > 0; i--)
	{
		// Troca o valor da posição i com o de uma posição aleatória
		uint j = rand() % (i + 1);
		swap(indexes[i], indexes[j]);
	}
}

//===========================================================================//

int MLP::getID() const
{
	return mlpID;
}

//===========================================================================//

void MLP::setID(int mlpID)
{
	this->mlpID = mlpID;
}

//===========================================================================//

string MLP::getName() const
{
	return name;
}

//===========================================================================//

void MLP::setName(string name)
{
	this->name = name;
}

//===========================================================================//

Range MLP::getRange() const
{
	return range;
}

//===========================================================================//

void MLP::setRange(Range range)
{
	this->range = range;
}

//===========================================================================//

uint MLP::getNLayers() const
{
	return layers.size();
}

//===========================================================================//

Layer& MLP::getLayer(uint i)
{
	return *(layers[i]);
}

//===========================================================================//

const Layer& MLP::getLayer(uint i) const
{
	return *(layers[i]);
}

//===========================================================================//

void MLP::setOutput()
{
	Layer *last = layers.back();
	output = last->getFuncSignal();
}

//===========================================================================//

}
