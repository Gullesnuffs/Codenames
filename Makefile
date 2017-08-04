CPP = src/*.cpp
FLAGS = -Wall -Wextra -Ofast -march=native -Wfatal-errors -std=c++11
all: $(CPP)
	g++ -o codenames $(FLAGS) $(CPP)
