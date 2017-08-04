#include "InappropriateEngine.h"
#include <fstream>

using namespace std;

InappropriateEngine::InappropriateEngine(const string &filePath, const Dictionary &dict) {
	load(filePath, dict);
}

void InappropriateEngine::load(const string &filePath, const Dictionary &dict) {
	ifstream fin(filePath);
	string s;
	inappropriateWords.resize(dict.size());
	while (getline(fin, s)) {
		s = normalize(s);
		if (dict.wordExists(s))
			inappropriateWords[dict.getID(s)] = true;

		// Pluralize!
		if (dict.wordExists(s + "s"))
			inappropriateWords[dict.getID(s + "s")] = true;

		// Adjectivize!
		if (dict.wordExists(s + "y"))
			inappropriateWords[dict.getID(s + "y")] = true;
	}
}

bool InappropriateEngine::isInappropriate(wordID id) const {
	return inappropriateWords[(int)id];
}