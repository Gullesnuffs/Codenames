#!/bin/bash
mkdir -p /tmp/codenames
for x in {1..10}
do
	cat test_data/ordered5/*.txt | ./codenames --extract-features /tmp/codenames/train${x}.txt /tmp/codenames/test${x}.txt ${x}
done
