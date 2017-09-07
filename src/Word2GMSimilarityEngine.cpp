#include "Word2GMSimilarityEngine.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define all(v) (v).begin(), (v).end()

using namespace std;

float Word2GMSimilarityEngine::similarity(const WordEmbedding &v1, const WordEmbedding &v2) {
	float sim = -1;
	float sum = 0;
	float cnt = 0;
	for(auto& g1 : v1.gaussians) {
		for(auto& g2 : v2.gaussians) {
			float dis = 0;
			rep(i,0,(int)g1.mus.size()){
				float diff = g1.mus[i] - g2.mus[i];
				dis += diff * diff;
			}
			cnt++;
			sum += 1/((dis+0.1)*(dis+0.1)*(dis+0.1));
			//sim = max(sim, (float)(-0.5 + 1.5 / (1.0 + 0.8 * dis * dis)));
		}
	}
	float mean = cbrt(cnt/sum) - 0.1;
	return -0.5 + 1.5 / (1.0 + 0.25 * mean * mean);
	//return sim;
}

/** Arbitrary statistic, in this case the word norm. */
float Word2GMSimilarityEngine::stat(wordID s) {
	return 1;
}

/** Returns true if successful */
bool Word2GMSimilarityEngine::load(const string &fileName, bool verbose) {
	cerr << "Load " << fileName << endl;
	int dimension, numberOfWords;
	modelid = formatVersion = 0;
	ifstream fin(fileName, ios::binary);
	fin.read((char *)&numberOfWords, sizeof numberOfWords);
	if (fin && numberOfWords == -1) {
		fin.read((char *)&formatVersion, sizeof formatVersion);
		fin.read((char *)&modelid, sizeof modelid);
		fin.read((char *)&numberOfWords, sizeof numberOfWords);
	}
	fin.read((char *)&dimension, sizeof dimension);
	if (!fin) {
		cerr << "Failed to load " << fileName << endl;
		return false;
	}
	if (verbose) {
		cerr << "Loading word2vec (" << numberOfWords << " words, " << dimension
			 << " dimensions, model " << modelid << '.' << formatVersion << ")... " << flush;
	}

	const int bufSize = 1 << 16;
	float norm;
	char buf[bufSize];
	string word;
	vector<float> values(dimension);
	vector<float> valuesd;
	// Note: Very conservative size, this may waste quite a lot of space if the words are already in
	// the dictionary
	words.resize(numberOfWords + dict.size());
	index2id.resize(numberOfWords);
	rep(i, 0, numberOfWords) {
		int len;
		fin.read((char *)&len, sizeof len);
		if (!fin) {
			cerr << "failed at reading entry " << i << endl;
			return false;
		}
		if (len > bufSize || len <= 0) {
			cerr << "invalid length " << len << endl;
			return false;
		}
		fin.read(buf, len);
		if (formatVersion >= 1) {
			fin.read((char *)&norm, sizeof norm);
		} else {
			norm = 1.0f;
		}
		fin.read((char *)values.data(), dimension * sizeof(float));
		if (!fin) {
			cerr << "failed at reading entry " << i << endl;
			return false;
		}
		word.assign(buf, buf + len);
		valuesd.assign(all(values));
		for(int i = 0; i < dimension; i++){
			valuesd[i] *= sqrt(norm);
		}
		wordID id = dict.addWord(word);
		words[id].gaussians.resize(2);
		words[id].gaussians[0].logsig = valuesd[0];
		words[id].gaussians[1].logsig = valuesd[dimension/2];
		words[id].gaussians[0].mus.resize(dimension/2-1);
		words[id].gaussians[1].mus.resize(dimension/2-1);
		for(int i = 0; i < dimension/2-1; i++){
			words[id].gaussians[0].mus[i] = valuesd[1+i];
			words[id].gaussians[1].mus[i] = valuesd[1+dimension/2+i];
		}
		index2id[i] = id;
	}
	if (verbose) {
		cerr << "done!" << endl;
	}
	return true;
}

bool Word2GMSimilarityEngine::wordExists(const string &word) {
	return dict.wordExists(word) && words[dict.getID(word)].gaussians.size() > 0;
}

float Word2GMSimilarityEngine::commutativeSimilarity(wordID fixedWord, wordID dynWord) {
	return similarity(words.at(fixedWord), words.at(dynWord));
}

float Word2GMSimilarityEngine::similarity(wordID fixedWord, wordID dynWord) {
	float sim = similarity(words.at(fixedWord), words.at(dynWord));
	return sim;
}
