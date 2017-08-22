#pragma once

#include "Dictionary.h"
#include "SimilarityEngine.h"

#include <map>
#include <string>
#include <vector>
#include <memory>

#include "Utilities.h"

struct MixingSimilarityEngine final : SimilarityEngine {
   public:
   	std::unique_ptr<SimilarityEngine> engine1;
   	std::unique_ptr<SimilarityEngine> engine2;
   	float multiplier1;
   	float multiplier2;

	MixingSimilarityEngine() {}

	/** Arbitrary statistic, in this case the word norm. */
	float stat(wordID s);

	/** Returns true if successful */
	bool load(const std::string &fileName, bool verbose);

	float similarity(wordID fixedWord, wordID dynWord);

	/** True if the word2vec model includes a vector for the specified word */
	bool wordExists(const std::string &word);
};
