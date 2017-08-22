#pragma once

#include "Dictionary.h"
#include "SimilarityEngine.h"

#include <map>
#include <string>
#include <vector>

#include "Utilities.h"

struct RandomSimilarityEngine final : SimilarityEngine {
   public:
	RandomSimilarityEngine() {}

	/** Arbitrary statistic, in this case the word norm. */
	float stat(wordID s);

	/** Returns true if successful */
	bool load(const std::string &fileName, bool verbose);

	float similarity(wordID fixedWord, wordID dynWord);
	float commutativeSimilarity(wordID word1, wordID word2);

	/** True if the word2vec model includes a vector for the specified word */
	bool wordExists(const std::string &word);
};
