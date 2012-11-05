#!/bin/sh

function exec_one_layer()
{
	./build/pmlp_serial fast 9 $1 3 150 1000 >> result/serial_one_$1.txt
}

function exec_two_layers()
{
	./build/pmlp_serial fast 9 $1 $2 3 150 1000 >> result/serial_two_$1_$2.txt
}

exec_one_layer 20
exec_one_layer 65
exec_one_layer 182
exec_one_layer 455
exec_one_layer 1040
exec_one_layer 2210
exec_one_layer 4420
exec_one_layer 8398
exec_one_layer 15270
exec_one_layer 26721
exec_one_layer 45220
exec_one_layer 74290
exec_one_layer 118864

exec_two_layers 14 5
exec_two_layers 26 17
exec_two_layers 43 45
exec_two_layers 69 61
exec_two_layers 106 97
exec_two_layers 154 146
exec_two_layers 219 211
exec_two_layers 302 294
exec_two_layers 408 400
exec_two_layers 541 532
exec_two_layers 704 695
exec_two_layers 902 894
exec_two_layers 1142 1133

