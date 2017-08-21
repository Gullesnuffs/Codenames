#include "EdgeListSimilarityEngine.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

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

float EdgeListSimilarityEngine::similarity(wordID fixedWord, wordID dynWord) {
	auto w = edges.find(make_pair(fixedWord, dynWord));
	float noise = rand() / (float)RAND_MAX;
	if (w != edges.end()) return w->second + noise*0.01f;
	else return 0 + noise*0.01f;
}
