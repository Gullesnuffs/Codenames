#!/bin/bash
results="0"
for x in {1..10}
do
	python3 training_tf.py train${x}.txt test${x}.txt results${x}.txt > /dev/null
	results="$results + `cat results${x}.txt`"
done
echo "Objective function: `echo $results | bc`"
