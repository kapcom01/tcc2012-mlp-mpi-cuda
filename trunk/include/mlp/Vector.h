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
	inline Vector()
	{
		this->data = NULL;
		this->vsize = 0;
		this->vstep = 1;
	}

	/**
	 * Constrói a partir de um vector
	 * @param vector Vetor localizado na memória da máquina
	 * @param step Tamanho do passo considerando o vetor como uma matriz
	 */
	__host__
	inline Vector(vector<T> &vector, uint step = 1)
	{
		data = &(vector[0]);
		vsize = vector.size();
		this->vstep = step;
	}

	/**
	 * Constrói a partir de um host_vector
	 * @param vector Vetor localizado na memória da máquina
	 * @param step Tamanho do passo considerando o vetor como uma matriz
	 */
	__host__
	inline Vector(host_vector<T> &vector, uint step = 1)
	{
		data = &(vector[0]);
		vsize = vector.size();
		this->vstep = step;
	}

	/**
	 * Contrói a partir de uma linha de uma matriz
	 * @param vector Vetor localizado na memória da máquina
	 * @param step Tamanho do passo da matriz
	 * @param index Índice da linha
	 */
	__host__
	inline Vector(host_vector<T> &vector, uint step, uint index,
			uint size, uint offset = 0)
	{
		this->data = &(vector[index * step + offset]);
		this->vsize = size;
		this->vstep = 1;
	}

	/**
	 * Constrói a partir de um host_vector
	 * @param vector Vetor localizado na memória do dispositivo
	 * @param step Tamanho do passo considerando o vetor como uma matriz
	 */
	__host__
	inline Vector(device_vector<T> &vector, uint step = 1)
	{
		data = raw_pointer_cast(&(vector[0]));
		vsize = vector.size();
		this->vstep = step;
	}

	/**
	 * Contrói a partir de uma linha de uma matriz
	 * @param vector Vetor localizado na memória do dispositivo
	 * @param step Tamanho do passo da matriz
	 * @param index Índice da linha
	 */
	__host__
	inline Vector(device_vector<T> &vector, uint step, uint index,
			uint size, uint offset = 0)
	{
		this->data = raw_pointer_cast(&(vector[index * step + offset]));
		this->vsize = size;
		this->vstep = 1;
	}

	/**
	 * Destrói o vetor
	 */
	__host__
	inline ~Vector()
	{

	}

	/**
	 * Atribui a este vetor os valores de outro vetor
	 * @param vec Outro vetor
	 */
	__host__
	inline void operator =(const Vector<T> &vec)
	{
		this->data = vec.data;
		this->vsize = vec.vsize;
		this->vstep = vec.vstep;
	}

	/**
	 * Retorna o tamanho do vetor
	 * @return Tamanho do vetor
	 */
	__host__ __device__
	inline uint size() const
	{
		return vsize;
	}

	/**
	 * Retorna o i-ésimo elemento do vetor
	 * @param i Índice do elemento
	 * @return i-ésimo elemento do vetor
	 */
	__host__ __device__
	inline T* operator ()(uint i)
	{
		return &(data[i * vstep]);
	}

	/**
	 * Retorna o i-ésimo elemento do vetor
	 * @param i Índice do elemento
	 * @return i-ésimo elemento do vetor
	 */
	__host__ __device__
	inline T operator [](uint i) const
	{
		return data[i];
	}

	/**
	 * Ponteiro para os dados
	 */
	T* data;

private:

	/**
	 * Tamanho do vetor
	 */
	uint vsize;

	/**
	 * Passo
	 */
	uint vstep;

};

/**
 * Vetor de double
 */
typedef Vector<float> vec_float;

/**
 * Vetor de Stat
 */
typedef Vector<Stat> vec_stat;

}

#endif
