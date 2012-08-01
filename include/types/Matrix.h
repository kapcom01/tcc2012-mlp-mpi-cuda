#ifndef MATRIX_H_
#define MATRIX_H_

#include "Common.h"

/**
 * Classe que representa uma matriz bidimensional
 */
template<class T>
class Matrix
{

public:

	/**
	 * Contrói uma nova matriz rows x cols
	 * @param rows Número de linhas
	 * @param cols Número de colunas
	 */
	Matrix(uint rows, uint cols = 1)
	{
		this->rows = rows, this->cols = cols;
		data = new T[rows * cols]();
	}

	/**
	 * Contrói uma nova matriz a partir de outra
	 * @param mat Matriz
	 */
	Matrix(const Matrix<T> &mat)
	{
		rows = mat.rows, cols = mat.cols;
		data = new T[rows * cols]();
		for (uint i = 0; i < size(); i++)
			data[i] = mat.data[i];
	}

	/**
	 * Destrói a matriz
	 */
	virtual ~Matrix()
	{
		delete[] data;
	}

	/**
	 * Retorna uma referência para o valor (i,j) da matriz
	 * @param i Linha
	 * @param j Coluna
	 * @return Referência para o valor requisitado
	 */
	T& at(uint i, uint j)
	{
		return data[i * cols + j];
	}

	/**
	 * Retorna uma referência constante para o valor (i,j) da matriz
	 * @param i Linha
	 * @param j Coluna
	 * @return Referência para o valor requisitado
	 */
	const T& at(uint i, uint j) const
	{
		return data[i * cols + j];
	}

	/**
	 * Retorna uma referência para o i-ésimo elemento da matriz
	 * @param i Elemento
	 * @return Referência para o elemento requisitado
	 */
	T& at(uint i)
	{
		return data[i];
	}

	/**
	 * Retorna uma referência constante para o i-ésimo elemento da matriz
	 * @param i Elemento
	 * @return Referência para o elemento requisitado
	 */
	const T& at(uint i) const
	{
		return data[i];
	}

	/**
	 * Retorna uma referência para o valor (i,j) da matriz
	 * @param i Linha
	 * @param j Coluna
	 * @return Referência para o valor requisitado
	 */
	T& operator ()(uint i, uint j)
	{
		return at(i, j);
	}

	/**
	 * Retorna uma referência constante para o valor (i,j) da matriz
	 * @param i Linha
	 * @param j Coluna
	 * @return Referência para o valor requisitado
	 */
	const T& operator ()(uint i, uint j) const
	{
		return at(i, j);
	}

	/**
	 * Retorna uma referência para o i-ésimo elemento da matriz
	 * @param i Elemento
	 * @return Referência para o elemento requisitado
	 */
	T& operator [](uint i)
	{
		return at(i);
	}

	/**
	 * Retorna uma referência constante para o i-ésimo elemento da matriz
	 * @param i Elemento
	 * @return Referência para o elemento requisitado
	 */
	const T& operator [](uint i)
	{
		return at(i);
	}

	/**
	 * Retorna um ponteiro para a linha i
	 * @param i Linha
	 * @return Ponteiro para a linha requisitada
	 */
	T* row(uint i)
	{
		return &(data[i * cols]);
	}

	/**
	 * Retorna um ponteiro constante para a linha i
	 * @param i Linha
	 * @return Ponteiro para a linha requisitada
	 */
	const T* row(uint i) const
	{
		return &(data[i * cols]);
	}

	/**
	 * Retorna o tamanho da matriz = linhas x colunas
	 * @return Tamanho da matriz
	 */
	ulong size() const
	{
		return rows * cols;
	}

	/**
	 * Linhas
	 */
	uint rows;

	/**
	 * Colunas
	 */
	uint cols;

protected:

	/**
	 * Vetor de valores
	 */
	T* data;

};

/**
 * Matriz de double
 */
typedef Matrix<double> MatrixLF;

/**
 * Ponteiro para Matriz de double
 */
typedef shared_ptr<MatrixLF> MatrixPtr;

/**
 * Vetor de números inteiros sem sinal
 */
typedef Matrix<uint> VectorU;

/**
 * Vetor de números flutuantes
 */
typedef Matrix<double> VectorLF;

/**
 * Ponteiro para vdouble
 */
typedef shared_ptr<VectorLF> VectorPtr;

#endif
