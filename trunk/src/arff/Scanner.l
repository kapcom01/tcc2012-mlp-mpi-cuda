%{
#include "arff/Scanner.h"
#include "arff/ParseException.h"

using namespace ARFF;

#define YY_NO_UNISTD_H

#undef YY_DECL
#define YY_DECL int Scanner::yylex(Parser::semantic_type* yylval, Parser::location_type* yylloc)

#define YY_USER_ACTION yylloc->columns(yyleng);

typedef Parser::token token;

%}

%option yylineno yyclass="Scanner" noyywrap nounput c++ batch

	/**
	 * Definições
	 */
C	[0-9A-Za-z_\-]
S1	[ -~]{-}["]
S2	[ -~]{-}[']
L	[a-zA-Z_]
D	[0-9]
WS	[ \r\t\n]
COM	[%]
ND	[!-~]{-}[,{}? ]

%%

%{
	// Reseta localização
	yylloc->step();
%}

	/**
	 * Regra de eliminação de espaços em brancos
	 */
{WS}+   {  }


	/**
	 * Regra de comentário por linha
	 */
{COM}.*   { }

	/**
	 * Regras de produção de palavras reservadas
	 */
@relation|@RELATION		{ return token::RELATION; }
@attribute|@ATTRIBUTE	{ return token::ATTRIBUTE; }
@data|@DATA				{ return token::DATA; }
numeric|NUMERIC			{ return token::TYPE_NUMERIC; }
real|REAL				{ return token::TYPE_NUMERIC; }
string|STRING			{ return token::TYPE_STRING; }
date|DATE				{ return token::TYPE_DATE; }

	/**
	 * Regras para produção de símbolos reservados
	 */
","	{ return token::COMMA; }
"{"	{ return token::RIGHT_BRACES; }
"}"	{ return token::LEFT_BRACES; }
"?"	{ return token::QUESTION; }

	/**
	 * Regras para produções de números inteiros
	 */
-?{D}+   {
	yylval->ival = atoi(yytext);
	return token::INTEGER;
}

	/**
	 * Regras para produções de números reais
	 */
(-?{D}*"."{D}+)|(-?{D}+"."{D}*)   {
	yylval->fval = atof(yytext);
	return token::REAL;
}

	/**
	 * Regra para produção de identificadores
	 */
{C}+   {
	yylval->sval = new string(yytext);
	return token::ID;
}

	/**
	 * Regras para produção de strings
	 */
(\"{S1}*\")|('{S2}*')   { 
	yylval->sval = new string(&(yytext[1]), strlen(yytext) - 2);
	return token::STRING;
}

	/**
	 * Regras para número mal formado: 1a1b.1, 1.a1, 1a.a1, 123a, 123@ ou 123ab123
	 */
({D}{ND}*"."{ND}*)|({D}{ND}*)   {
	throwError(LEX_MALFORMED_NUMBER);
}

	/**
	 * Regra para aspas duplas não fechadas: "absa
	 */
(\"{S1}*$)|('{S2}*$)   { 
	throwError(LEX_QUOTATION_NOT_CLOSED);
}

	/**
	 * Regras para erros de símbolos desconhecidos
	 */
.	   {
	throwError(LEX_UNKNOWN_TOKEN);
}

