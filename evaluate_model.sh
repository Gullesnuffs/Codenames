#!/bin/bash
results="0"

parallel -i bash -c "python3 training_tf.py /tmp/codenames/train{}.txt /tmp/codenames/test{}.txt /tmp/codenames/results{}.txt &>/dev/null" -- {1..10}

for x in {1..10}; do
	results="$results + `cat /tmp/codenames/results${x}.txt`"
	echo "Objective function: `echo $results | bc`"
done
echo "Objective function: `echo $results | bc`"
