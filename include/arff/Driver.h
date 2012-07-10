#ifndef DRIVER_H_
#define DRIVER_H_

#include "Defines.h"

namespace ARFF
{

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
    class DataSet* parse();

    /**
     * Analisador léxico
     */
    class Scanner* scanner;

    /**
     * Conjunto de dados
     */
    class DataSet* dataset;

    friend class Scanner;
    friend class Parser;

private:

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
    class Parser* parser;

};

/**
 * Ponteiro inteligente para Driver
 */
typedef shared_ptr<Driver> DriverPtr;

}

#endif
