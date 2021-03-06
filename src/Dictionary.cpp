#include "Dictionary.h"
#include <fstream>

using namespace std;

string normalize(string s) {
	for (auto& c : s) {
		if ('A' <= c && c <= 'Z') {
			c = (char)tolower(c);
		} else if (c == ' ') {
			c = '_';
		}
	}
	return s;
}

string denormalize(string s) {
	for (auto& c : s) {
		if (c == '_') {
			c = ' ';
		}
	}
	return s;
}

bool superOrSubstring(const string& a, const string& b) {
	auto lowerA = normalize(a);
	auto lowerB = normalize(b);
	return lowerA.find(lowerB) != string::npos || lowerB.find(lowerA) != string::npos;
}

int Dictionary::getPopularity(wordID id) const {
	// Word IDs are the indices of words in the input file, which is assumed to be ordered
	// according to popularity
	return id + 1;
}

bool Dictionary::wordExists(const string& word) const {
	return word2id.count(word) > 0;
}

wordID Dictionary::addWord(const string& word) {
	if (!wordExists(word)) {
		word2id.insert(make_pair(word, (wordID)words.size()));
		words.push_back(word);
	}
	return getID(word);
}

string& Dictionary::getWord(wordID id) {
	return words[(int)id];
}

wordID Dictionary::getID(const string& word) const {
	return word2id.at(word);
}

/** Top N most popular words */
vector<wordID> Dictionary::getCommonWords(int vocabularySize) const {
	vector<wordID> ret;
	vocabularySize = min(vocabularySize, (int)words.size());
	ret.reserve(vocabularySize);
	for (int i = 0; i < vocabularySize; i++) {
		ret.push_back(wordID(i));
	}
	return ret;
}