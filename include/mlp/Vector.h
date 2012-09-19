#ifndef VECTOR_H_
#define VECTOR_H_

#include "mlp/Types.h"

namespace ParallelMLP
{

/**
 * Classe que armazena um ponteiro para os dados de um vetor e seu tamanho
 */
template<class T> class Vector
{

public:


	/**
	 * Construtor padrão
	 */
	__host__
	Vector()
	{
		this->data = NULL;
		this->size = 0;
		this->step = 1;
	}

	/**
	 * Constrói a partir de um ponteiro para um vetor
	 * @param data Ponteiro para um vetor
	 * @param size Tamanho do vetor
	 * @param step Tamanho do passo considerando o vetor como uma matriz
	 */
	__host__
	Vector(T* data, uint size, uint step = 1)
	{
		this->data = data;
		this->size = size;
		this->step = step;
	}

	/**
	 * Constrói a partir de um vector
	 * @param vector Vetor localizado na memória da máquina
	 * @param step Tamanho do passo considerando o vetor como uma matriz
	 */
	__host__
	Vector(vector<T> &vector, uint step = 1)
	{
		data = &(vector[0]);
		size = vector.size();
		this->step = step;
	}

	/**
	 * Constrói a partir de um host_vector
	 * @param vector Vetor localizado na memória da máquina
	 * @param step Tamanho do passo considerando o vetor como uma matriz
	 */
	__host__
	Vector(host_vector<T> &vector, uint step = 1)
	{
		data = &(vector[0]);
		size = vector.size();
		this->step = step;
	}

	/**
	 * Constrói a partir de um host_vector
	 * @param vector Vetor localizado na memória do dispositivo
	 * @param step Tamanho do passo considerando o vetor como uma matriz
	 */
	__host__
	Vector(device_vector<T> &vector, uint step = 1)
	{
		data = vector.data().get();
		size = vector.size();
		this->step = step;
	}

	/**
	 * Destrói o vetor
	 */
	__host__
	~Vector()
	{

	}

	/**
	 * Atribui a este vetor os valores de outro vetor
	 * @param vec Outro vetor
	 */
	__host__
	void operator =(const Vector<T> &vec)
	{
		this->data = vec.data;
		this->size = vec.size;
		this->step = vec.step;
	}

	/**
	 * Retorna o i-ésimo elemento do vetor
	 * @param i Índice do elemento
	 * @return i-ésimo elemento do vetor
	 */
	__host__ __device__
	T* operator [](uint i) const
	{
		return &(data[i * step]);
	}

	/**
	 * Ponteiro para os dados
	 */
	T* data;

	/**
	 * Tamanho do vetor
	 */
	uint size;

	/**
	 * Passo
	 */
	uint step;

};

}

#endif
