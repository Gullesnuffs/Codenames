#!/bin/bash
for x in {1..10}
do
	cat test_data/ordered5/*.txt | ./codenames --extract-features train${x}.txt test${x}.txt ${x}
done
