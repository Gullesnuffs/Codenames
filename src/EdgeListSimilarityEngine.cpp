#include "EdgeListSimilarityEngine.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define all(v) (v).begin(), (v).end()

using namespace std;

/** Arbitrary statistic */
float EdgeListSimilarityEngine::stat(wordID s) {
	return 0;
}

/** Returns true if successful */
bool EdgeListSimilarityEngine::load(const string &fileName, bool verbose) {
	ifstream fin(fileName);
	if (!fin) return false;

	int numEdges;
	fin >> numEdges;
	for (int i = 0; i < numEdges; i++) {
		string a, b;
		float weight;
		fin >> a >> b >> weight;
		if (dict.wordExists(a) && dict.wordExists(b)) {
			edges[make_pair(dict.getID(a), dict.getID(b))] = weight;
			edges[make_pair(dict.getID(b), dict.getID(a))] = weight;
		}
	}
	return true;
}

bool EdgeListSimilarityEngine::wordExists(const string &word) {
	return dict.wordExists(word);
}

float EdgeListSimilarityEngine::commutativeSimilarity(wordID word1, wordID word2) {
	auto w = edges.find(make_pair(word1, word2));
	float noise = rand() / (float)RAND_MAX;
	if (w != edges.end()) return w->second + noise*0.00f;
	else return 0 + noise*0.00f;
}

float EdgeListSimilarityEngine::similarity(wordID fixedWord, wordID dynWord) {
	return commutativeSimilarity(fixedWord, dynWord);
}
