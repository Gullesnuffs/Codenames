#pragma once

#include "Dictionary.h"
#include "SimilarityEngine.h"

#include <string>
#include <vector>
#include <map>

#include "Utilities.h"

struct Word2VecSimilarityEngine final : SimilarityEngine {
   private:
	int formatVersion, modelid;
	std::map<std::string, wordID> word2id;
	std::vector<std::vector<float>> words;
	std::vector<std::string> wordsStrings;
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
	/** Arbitrary statistic, in this case the word norm. */
	float stat(wordID s);

	std::vector<float> getVector(wordID s);

	/** Returns true if successful */
	bool load(const std::string &fileName, bool verbose);

	/** Top N most popular words */
	std::vector<wordID> getCommonWords(int vocabularySize);

	float similarity(wordID fixedWord, wordID dynWord);

	/** ID representing a particular word */
	wordID getID(const std::string &s);

	/** Popularity of a word, the most popular word has a popularity of 1, the second most popular
	 * has a popularity of 2 etc. */
	int getPopularity(wordID id);

	/** Word string corresponding to the ID */
	const std::string &getWord(wordID id);

	/** True if the word2vec model includes a vector for the specified word */
	bool wordExists(const std::string &word);

	std::vector<std::pair<float, std::string>> similarWords(const std::string &s);
};
