#!/bin/bash
if [ $1 -eq 1 ]
then
    python3 "Q1/$4.py" "$2" "$3"
fi

if [ $1 -eq 2 ]
then
    if [ $4 -eq 0 ]
    then
        python3 "Q2/binary_$5.py" "$2" "$3"
    fi
    if [ $4 -eq 1 ]
    then
        python3 "Q2/multi_$5.py" "$2" "$3"
    fi
fi
