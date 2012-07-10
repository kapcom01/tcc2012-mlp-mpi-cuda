#include "arff/Driver.h"
#include "arff/Scanner.h"
#include "arff/Parser.hh"
#include "arff/DataSet.h"

namespace ARFF
{

/**
 * Constrói um driver a partir de um arquivo ARFF
 * @param filename Nome do arquivo ARFF
 */
Driver::Driver(const string &filename)
{
    this->filename = filename;

    istream.open(filename.c_str());
    if(!istream.good())
    {
        cerr << "Couldn't open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    dataset = new DataSet(*this);
    scanner = new Scanner(*this);
    parser = new Parser(*this);
}

/**
 * Destrói o driver
 */
Driver::~Driver()
{
    delete dataset;
    delete scanner;
    delete parser;
}

/**
 * Realiza o parseamento
 * @return Retorna o conjunto de dados contido no arquivo
 */
DataSet* Driver::parse()
{
    parser->parse();
    return dataset;
}

}
