#pragma once

#include <string>
#include <vector>
#include "Dictionary.h"

struct SimilarityEngine {
	virtual bool load(const std::string &fileName, bool verbose) = 0;
	virtual float similarity(wordID fixedWord, wordID dynWord) = 0;

	/** A commutative similarity measure, in contrast to the #similarity function which may change depending on the order of the parameters */
	virtual float commutativeSimilarity(wordID word1, wordID word2) = 0;
	virtual bool wordExists(const std::string &word) = 0;
	virtual float stat(wordID s) = 0;
	virtual ~SimilarityEngine() {}
};