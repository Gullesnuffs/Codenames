#pragma once

#include "Dictionary.h"
#include "SimilarityEngine.h"

#include <map>
#include <string>
#include <vector>

#include "Utilities.h"

struct Word2VecSimilarityEngine final : SimilarityEngine {
   private:
	int formatVersion, modelid;
	std::vector<std::vector<float>> words;
	std::vector<wordID> index2id;
	Dictionary &dict;
	enum Models { GLOVE = 1, CONCEPTNET = 2 };

	// All word vectors are stored normalized -- wordNorms holds their original squared norms.
	// In some embeddings, words that have more (specific) meanings have higher norms.
	std::vector<float> wordNorms;

	/** Similarity between two word vectors.
	 * Implemented as an inner product. This is the main bottleneck of the
	 * engine, and it gains a lot from being compiled with "-O3 -mavx".
	 */
	float similarity(const std::vector<float> &v1, const std::vector<float> &v2);

   public:
	inline int dimension() {
		// TODO: Use field instead?
		return words[0].size();
	}

	Word2VecSimilarityEngine(Dictionary &dict) : dict(dict) {}

	/** Arbitrary statistic, in this case the word norm. */
	float stat(wordID s);

	std::vector<float> getVector(wordID s);

	inline float getNorm(wordID s) {
		return wordNorms[(int)s];
	}

	/** Returns true if successful */
	bool load(const std::string &fileName, bool verbose);

	float commutativeSimilarity(wordID word1, wordID word2);
	float similarity(wordID fixedWord, wordID dynWord);

	/** True if the word2vec model includes a vector for the specified word */
	bool wordExists(const std::string &word);

	std::vector<std::pair<float, std::string>> similarWords(const std::string &s);
	std::vector<std::pair<float, std::string>> similarWords(const std::vector<float> &vec);
};
