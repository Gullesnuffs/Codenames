#include "Word2VecSimilarityEngine.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define all(v) (v).begin(), (v).end()

using namespace std;

float Word2VecSimilarityEngine::similarity(const vector<float> &v1, const vector<float> &v2) {
	float ret = 0;
	int dim = (int)v1.size();
	rep(i, 0, dim) {
		ret += v1[i] * v2[i];
	}
	return ret;
}

/** Arbitrary statistic, in this case the word norm. */
float Word2VecSimilarityEngine::stat(wordID s) {
	return wordNorms[s];
}

vector<float> Word2VecSimilarityEngine::getVector(wordID s) {
	return words[s];
}

/** Returns true if successful */
bool Word2VecSimilarityEngine::load(const string &fileName, bool verbose) {
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
	words.resize(numberOfWords);
	wordsStrings.resize(numberOfWords);
	wordNorms.resize(numberOfWords);
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
		words[i] = move(valuesd);
		wordsStrings[i] = word;
		word2id[word] = wordID(i);
		wordNorms[i] = norm;
		if (modelid == Models::GLOVE) {
			wordNorms[i] = min(pow(wordNorms[i], 0.4f), 5.3f);
		}
	}
	if (verbose) {
		cerr << "done!" << endl;
	}
	return true;
}

/** Top N most popular words */
vector<wordID> Word2VecSimilarityEngine::getCommonWords(int vocabularySize) {
	vector<wordID> ret;
	vocabularySize = min(vocabularySize, (int)words.size());
	ret.reserve(vocabularySize);
	for (int i = 0; i < vocabularySize; i++) {
		ret.push_back(wordID(i));
	}
	return ret;
}

float Word2VecSimilarityEngine::similarity(wordID fixedWord, wordID dynWord) {
	float sim = similarity(words[fixedWord], words[dynWord]);
	if (modelid == Models::GLOVE) {
		return sim * wordNorms[dynWord] / 4.5f;
	} else if (modelid == Models::CONCEPTNET) {
		return (sim <= 0 ? sim : pow(sim, 0.8f) * 1.6f);
	} else {
		return sim;
	}
}

/** ID representing a particular word */
wordID Word2VecSimilarityEngine::getID(const string &s) {
	return word2id.at(s);
}

/** Popularity of a word, the most popular word has a popularity of 1, the second most popular
 * has a popularity of 2 etc. */
int Word2VecSimilarityEngine::getPopularity(wordID id) {
	// Word IDs are the indices of words in the input file, which is assumed to be ordered
	// according to popularity
	return id + 1;
}

/** Word string corresponding to the ID */
const string& Word2VecSimilarityEngine::getWord(wordID id) {
	return wordsStrings[id];
}

/** True if the word2vec model includes a vector for the specified word */
bool Word2VecSimilarityEngine::wordExists(const string &word) {
	return word2id.count(word) > 0;
}

vector<pair<float, string>> Word2VecSimilarityEngine::similarWords(const string &s) {
	if (!wordExists(s)) {
		cout << denormalize(s) << " does not occur in the corpus" << endl;
		return vector<pair<float, string>>();
	}
	vector<pair<float, wordID>> ret;
	rep(i, 0, (int)words.size()) {
		ret.push_back(make_pair(-similarity(getID(s), wordID(i)), wordID(i)));
	}
	sort(all(ret));
	vector<pair<float, string>> res;
	rep(i, 0, 10) {
		res.push_back(make_pair(-ret[i].first, getWord(ret[i].second)));
	}
	return res;
}
