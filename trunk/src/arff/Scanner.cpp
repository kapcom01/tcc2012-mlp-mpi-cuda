#include "arff/Scanner.h"

namespace ARFF
{

//===========================================================================//

Scanner::Scanner(Driver &cDriver)
		: yyFlexLexer(&(cDriver.istream)), driver(cDriver)
{

}

//===========================================================================//

void Scanner::markLine()
{
	markedLine = yylineno;
}

//===========================================================================//

int Scanner::getLineno() const
{
	return markedLine;
}

//===========================================================================//

string Scanner::getToken() const
{
	return yytext;
}

//===========================================================================//

}