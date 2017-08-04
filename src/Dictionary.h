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
	Dictionary() {}
	Dictionary(const std::string& filePath);

	/** Popularity of a word, the most popular word has a popularity of 1, the second most popular
	 * has a popularity of 2 etc. */
	int getPopularity(wordID id) const;

	/** True if the dictionary includes the specified word */
	bool wordExists(const std::string& word) const;

	/** Word string corresponding to the ID */
	std::string& getWord(wordID id);

	/** ID representing a particular word */
	wordID getID(const std::string& word) const;

	/** Top N most popular words */
	std::vector<wordID> getCommonWords(int vocabularySize) const;

	inline int size() const {
		return (int)words.size();
	}

	/** Load the dictionary from a file.
	 * The file should be a '\n' separated list of words
	 * ordered according to their popularity (starting with the most popular word).
	 */
	void load(const std::string& filePath);
};