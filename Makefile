CPP = src/*.cpp
H = src/*.h
FLAGS = -Wall -Wextra -Ofast -march=native -Wfatal-errors -std=c++11

all: codenames calc

codenames: $(CPP) $(H)
	g++ -o codenames $(FLAGS) $(CPP)

calc: calc.cpp
	g++ -o calc $(FLAGS) calc.cpp

format:
	clang-format -i $(CPP) $(H)
