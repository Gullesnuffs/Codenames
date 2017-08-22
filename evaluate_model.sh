#!/bin/bash
results="0"
for x in {1..10}; do
	python3 training_tf.py /tmp/codenames/train${x}.txt /tmp/codenames/test${x}.txt /tmp/codenames/results${x}.txt > /dev/null &
done

wait

for x in {1..10}; do
	results="$results + `cat /tmp/results${x}.txt`"
	echo "Objective function: `echo $results | bc`"
done
echo "Objective function: `echo $results | bc`"
