#pragma once

#include "Dictionary.h"
#include "SimilarityEngine.h"

#include <map>
#include <string>
#include <vector>

#include "Utilities.h"

struct Gaussian {
	std::vector<float> mus;
	float logsig;
};

struct WordEmbedding{
	std::vector<Gaussian> gaussians;
};

struct Word2GMSimilarityEngine final : SimilarityEngine {
   private:
	int formatVersion, modelid;
	std::vector<WordEmbedding> words;
	std::vector<wordID> index2id;
	Dictionary &dict;
	float similarity(const WordEmbedding &v1, const WordEmbedding &v2);
	enum Models { GLOVE = 1, CONCEPTNET = 2, WORD2GM = 3 };

   public:

	Word2GMSimilarityEngine(Dictionary &dict) : dict(dict) {}

	/** Arbitrary statistic, in this case the word norm. */
	float stat(wordID s);

	/** Returns true if successful */
	bool load(const std::string &fileName, bool verbose);

	float commutativeSimilarity(wordID word1, wordID word2);
	float similarity(wordID fixedWord, wordID dynWord);

	/** True if the word2vec model includes a vector for the specified word */
	bool wordExists(const std::string &word);
};
