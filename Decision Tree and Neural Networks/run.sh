#!/bin/bash
if [ $1 -eq 1 ]
then
    python3 "Q1/$5.py" "$2" "$3" "$4"
fi

if [ $1 -eq 2 ]
then
    python3 "Q2/$4.py" "$2" "$3"
fi
