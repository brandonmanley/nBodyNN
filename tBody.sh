#! /usr/bin/env bash

./Brutus/main.exe 1 > outputBrut.txt & 
./threeBody.wls 1
# python syncData.py 