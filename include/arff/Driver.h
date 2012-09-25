#ifndef DRIVER_H_
#define DRIVER_H_

#include "Common.h"

namespace ParallelMLP
{

class Relation;
class Scanner;
class Parser;

/**
 * Classe responsável por fornecer uma camada de abstração do arquivo ARFF
 */
class Driver
{

public:

	/**
	 * Constrói um driver a partir de um arquivo ARFF
	 * @param filename Nome do arquivo ARFF
	 */
	Driver(const string &filename);

	/**
	 * Destrói o driver
	 */
	virtual ~Driver();

	/**
	 * Realiza o parseamento
	 * @return Retorna o conjunto de dados contido no arquivo
	 */
	Relation* parse();

	/**
	 * Analisador léxico
	 */
	Scanner* scanner;

	/**
	 * Conjunto de dados
	 */
	Relation* dataset;

	/**
	 * Nome do arquivo de entrada
	 */
	string filename;

	/**
	 * Stream de entrada
	 */
	ifstream istream;

	/**
	 * Analisador sintático
	 */
	Parser* parser;

};

/**
 * Ponteiro inteligente para Driver
 */
typedef shared_ptr<Driver> DriverPtr;

}

#endif
