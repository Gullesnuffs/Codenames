#include "RandomSimilarityEngine.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define all(v) (v).begin(), (v).end()

using namespace std;

/** Arbitrary statistic */
float RandomSimilarityEngine::stat(wordID s) {
	return 0;
}

/** Returns true if successful */
bool RandomSimilarityEngine::load(const string &fileName, bool verbose) {
	return true;
}

bool RandomSimilarityEngine::wordExists(const string &word) {
	return true;
}

float RandomSimilarityEngine::similarity(wordID fixedWord, wordID dynWord) {
	return rand() / (float)RAND_MAX;
}
