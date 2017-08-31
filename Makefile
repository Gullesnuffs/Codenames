H = src/*.h
FLAGS = -Wall -Wextra -Ofast -march=native -Wfatal-errors -std=c++11

all: codenames calc

COMMON_CPP = src/Bot.cpp src/EdgeListSimilarityEngine.cpp src/MixingSimilarityEngine.cpp src/RandomSimilarityEngine.cpp src/ProbabilityBot.cpp src/FuzzyBot.cpp src/Dictionary.cpp src/GameInterface.cpp src/InappropriateEngine.cpp src/Utilities.cpp src/Word2VecSimilarityEngine.cpp

codenames: $(H) $(COMMON_CPP) src/codenames.cpp
	g++ -o codenames $(FLAGS) $(COMMON_CPP) src/codenames.cpp

calc: $(H) $(COMMON_CPP) src/calc.cpp
	g++ -o calc $(FLAGS) $(COMMON_CPP) src/calc.cpp

preprocess: preprocess.cpp
	g++ -o preprocess $(FLAGS) preprocess.cpp

format:
	clang-format -style=file -i src/*.cpp $(H)
