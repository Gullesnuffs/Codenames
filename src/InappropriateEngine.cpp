#include "InappropriateEngine.h"
#include "Dictionary.h"
#include <fstream>

using namespace std;

void InappropriateEngine::load(string fileName) {
	ifstream fin(fileName);
	string s;
	while (getline(fin, s)) {
		inappropriateWords.insert(s);
		// Pluralize!
		inappropriateWords.insert(s + "s");
		// Adjectivize!
		inappropriateWords.insert(s + "y");
	}
}

bool InappropriateEngine::isInappropriate(string word) {
	return inappropriateWords.count(normalize(word));
}