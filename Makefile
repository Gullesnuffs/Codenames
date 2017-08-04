CPP = src/*.cpp
H = src/*.h
FLAGS = -Wall -Wextra -Ofast -march=native -Wfatal-errors -std=c++11

all: codenames dictionary.txt calc

codenames: $(CPP) $(H)
	g++ -o codenames $(FLAGS) $(CPP)

calc: calc.cpp
	g++ -o calc $(FLAGS) calc.cpp

dictionary.txt: data.txt
	echo "Generating dictionary file..."
	cat data.txt | awk "{ print $$1 }" > dictionary.txt

format:
	clang-format -i $(CPP) $(H)
