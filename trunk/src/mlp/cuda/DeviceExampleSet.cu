#include "mlp/cuda/DeviceExampleSet.h"

namespace ParallelMLP
{

__host__ __device__
void adjust(float* x, const Range* from, const Range* to);

//===========================================================================//

DeviceExampleSet::DeviceExampleSet(int relationID, int mlpID, SetType type)
	: ExampleSet(relationID, mlpID, type)
{

}

//===========================================================================//

DeviceExampleSet::~DeviceExampleSet()
{

}

//===========================================================================//

__global__
void normalizeVec(Vector<float> vec, Vector<Stat> stat)
{
	int k = blockIdx.x;
	int i = threadIdx.x;

	adjust(&(vec[k][i]), &(stat[i]->from), &(stat[i]->to));
}

//===========================================================================//

__global__
void unnormalizeVec(Vector<float> vec, Vector<Stat> stat)
{
	int k = blockIdx.x;
	int i = threadIdx.x;

	adjust(&(vec[k][i]), &(stat[i]->to), &(stat[i]->from));
}

//===========================================================================//

void DeviceExampleSet::copyToDevice()
{
	// Copia os dados da memória para a GPU
	devInput = input;
	devTarget = target;
	devInStat = inStat;
	devOutStat = outStat;

	// Atribui os ponteiros puros
	rawInput = Vector<float>(devInput, inVars);
	rawTarget = Vector<float>(devTarget, outVars);
	rawInStat = Vector<Stat>(devInStat);
	rawOutStat = Vector<Stat>(devOutStat);
}

//===========================================================================//

void DeviceExampleSet::copyToHost()
{
	// Copia os dados da GPU para a memória
	input = devInput;
	target = devTarget;
}

//===========================================================================//

void DeviceExampleSet::normalize()
{
	if (isNormalized)
		return;

	cout << endl << "=========== Before" << endl;
	print();

	// Copia os dados para o dispositivo
	copyToDevice();

	// Normaliza as colunas de entrada
	normalizeVec<<<size, inVars>>>(rawInput, rawInStat);

	// Normaliza as colunas de saída alvo
	normalizeVec<<<size, outVars>>>(rawTarget, rawOutStat);

	cout << devInput.size() << " " << devInput[0] << endl;

	// Copia os dados de volta para a memória
	copyToHost();

	cout << endl << "=========== After" << endl;
	print();

	isNormalized = true;
}

//===========================================================================//

void DeviceExampleSet::unnormalize()
{
	if (!isNormalized)
		return;

	cout << endl << "=========== Before" << endl;
	print();

	// Normaliza as colunas de entrada
	unnormalizeVec<<<size, inVars>>>(rawInput, rawInStat);

	// Normaliza as colunas de saída alvo
	unnormalizeVec<<<size, outVars>>>(rawTarget, rawOutStat);

//	 Normaliza as colunas de saída da rede neural
//	unnormalizeVec<<<size, outVars>>>(rawOutput, rawOutStat, outVars);

	// Copia os dados de volta para a memória
	copyToHost();

	cout << endl << "=========== After" << endl;
	print();

	isNormalized = false;
}

//===========================================================================//

__host__ __device__
void adjust(float* x, const Range* from, const Range* to)
{
	*x = (to->upper - to->lower) / (from->upper - from->lower)
			* (*x - from->lower) + to->lower;
}

//===========================================================================//

}
