#!/bin/bash
cat test_data/ordered5/*.txt | ./codenames --extract-features train.txt test.txt
