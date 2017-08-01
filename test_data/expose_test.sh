#!/bin/bash
cd ..
tmppipe=$(mktemp -u)
mkfifo "$tmppipe"
echo "Starting..."
./codenames --server 0<"$tmppipe" | nc -l 0.0.0.0 $1 1>"$tmppipe"