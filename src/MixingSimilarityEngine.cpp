#include "MixingSimilarityEngine.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define all(v) (v).begin(), (v).end()

using namespace std;

/** Arbitrary statistic */
float MixingSimilarityEngine::stat(wordID s) {
	return engine1->stat(s);
}

/** Returns true if successful */
bool MixingSimilarityEngine::load(const string &fileName, bool verbose) {
	return true;
}

bool MixingSimilarityEngine::wordExists(const string &word) {
	return engine1->wordExists(word) && engine2->wordExists(word);
}

float MixingSimilarityEngine::similarity(wordID fixedWord, wordID dynWord) {
	return engine1->similarity(fixedWord, dynWord) * multiplier1 + engine2->similarity(fixedWord, dynWord) * multiplier2;
}

float MixingSimilarityEngine::commutativeSimilarity(wordID word1, wordID word2) {
	return engine1->commutativeSimilarity(word1, word2) * multiplier1 + engine2->commutativeSimilarity(word1, word2) * multiplier2;
}
