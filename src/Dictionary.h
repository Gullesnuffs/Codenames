#pragma once

#include <map>
#include <string>
#include <vector>

/** Represents a single word or phrase in a similarity engine */
enum wordID : int {};

std::string normalize(std::string s);

std::string denormalize(std::string s);

/** True if a is a super or substring of b or vice versa */
bool superOrSubstring(const std::string& a, const std::string& b);

struct Dictionary {
   private:
	std::map<std::string, wordID> word2id;
	std::vector<std::string> words;

   public:
	/** Popularity of a word, the most popular word has a popularity of 1, the second most popular
	 * has a popularity of 2 etc. */
	int getPopularity(wordID id) const;

	/** True if the dictionary includes the specified word */
	bool wordExists(const std::string& word) const;

	/** Adds a word to the dictionary unless it already exists.
	 * Returns the ID of the word (regardless of whether it was already in the dictionary or not).
	 */
	wordID addWord(const std::string& word);

	/** Word string corresponding to the ID */
	std::string& getWord(wordID id);

	/** ID representing a particular word */
	wordID getID(const std::string& word) const;

	/** Top N most popular words */
	std::vector<wordID> getCommonWords(int vocabularySize) const;

	inline int size() const {
		return (int)words.size();
	}
};