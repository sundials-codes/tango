#!/bin/bash

PAR1='alpha'
#PAR1R=$(seq 0 5 30)
PAR1R='0.3 0.4'
PAR2='depth'
PAR2R=$(seq 0 3)
EXTRA=

for i in $PAR1R; do
	for j in $PAR2R; do
		IT=`python3 kinsol_simple.py $PAR1 $i $PAR2 $j $EXTRA`
		echo "($i,$j) $IT"
	done
done
