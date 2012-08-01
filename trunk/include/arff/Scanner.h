#ifndef SCANNER_H_
#define SCANNER_H_

// Usado para definir o include apenas uma vez
#if ! defined(yyFlexLexerOnce)
#include <FlexLexer.h>
#endif

#include "arff/Parser.hh"

namespace ARFF
{

/**
 * Classe responsável por extrair tokens do arquivo ARFF
 */
class Scanner: public yyFlexLexer
{

public:

	/**
	 * Contrói um scanner passando um driver
	 * @param cDriver Driver
	 */
	Scanner(Driver &cDriver)
			: yyFlexLexer(&(cDriver.istream)), driver(cDriver)
	{
	}

	/**
	 * Função de escaneamento com valor semântico
	 * @param yylval Valor semântico
	 * @param yylloc Localização no arquivo
	 * @return Token
	 */
	int yylex(Parser::semantic_type* yylval, Parser::location_type* yylloc);

	/**
	 * Marca a linha atual
	 */
	void markLine()
	{
		markedLine = yylineno;
	}

	/**
	 * Retorna a linha atual
	 * @return Linha atual
	 */
	int getLineno() const
	{
		return markedLine;
	}

	/**
	 * Retorna o token atual
	 * @return Token atual
	 */
	string getToken() const
	{
		return yytext;
	}

private:

	/**
	 * Driver
	 */
	Driver &driver;

	/**
	 * Linha marcada
	 */
	int markedLine;

};

}

#endif
