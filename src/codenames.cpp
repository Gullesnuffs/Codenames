#include "Bot.h"
#include "Dictionary.h"
#include "GameInterface.h"
#include "InappropriateEngine.h"
#include "SimilarityEngine.h"
#include "Utilities.h"
#include "Word2VecSimilarityEngine.h"
#include "EdgeListSimilarityEngine.h"
#include "MixingSimilarityEngine.h"
#include "RandomSimilarityEngine.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define rrep(i, a, b) for (int i = (a)-1; i >= int(b); --i)
#define trav(x, v) for (auto &x : v)
#define all(v) (v).begin(), (v).end()

using namespace std;

string escapeJSON(const string &s) {
	string res;
	auto hex = [](unsigned int c) -> char {
		if (c < 10)
			return (char)('0' + c);
		else
			return (char)('a' + c - 10);
	};
	trav(ch, s) {
		unsigned char c = (unsigned char)ch;
		if (c < 32 || c == 0x7f || c == '\\' || c == '"' || c == '/') {
			res += "\\u00";
			res += hex(c / 16);
			res += hex(c % 16);
		} else {
			res += ch;
		}
	}
	return res;
}

void batchMain() {
	typedef Bot::CardType CardType;
	typedef Bot::Difficulty Difficulty;
	auto fail = [](const char *message) {
		cout << "{\"status\": 0, \"message\": \"" << message << "\"}";
		exit(0);
	};

	try {
		cin.exceptions(ios::failbit | ios::eofbit | ios::badbit);

		string engine;
		cin >> engine;
		if (engine == "glove")
			engine = "models/glove.840B.330d.bin";
		else if (engine == "conceptnet")
			engine = "models/conceptnet.bin";
		else if (engine == "conceptnet-swe")
			engine = "models/conceptnet-swedish.bin";
		else
			fail("Invalid engine parameter.");

		Dictionary dict;
		Word2VecSimilarityEngine word2vecEngine(dict);
		if (!word2vecEngine.load(engine, false))
			fail("Unable to load similarity engine.");

		InappropriateEngine inappropriateEngine("inappropriate.txt", dict);
		ProbabilityBot bot(dict, word2vecEngine, inappropriateEngine);

		char color;
		cin >> color;
		if (color != 'r' && color != 'b')
			fail("Invalid color.");

		string type;
		while (cin >> type && type != "go") {
			if (type == "hinted") {
				string word;
				cin >> word;
				bot.setHasInfo(word);
				continue;
			}
			if (type == "clue") {
				string word;
				cin >> word;
				bot.addOldClue(word);
				continue;
			}
			if (type == "difficulty") {
				string difficulty;
				cin >> difficulty;
				Difficulty diff;
				if (difficulty == "easy")
					diff = Difficulty::EASY;
				else if (difficulty == "medium")
					diff = Difficulty::MEDIUM;
				else if (difficulty == "hard")
					diff = Difficulty::HARD;
				else {
					fail("Invalid difficulty.");
					abort();
				}
				bot.setDifficulty(diff);
				continue;
			}
			if (type == "inappropriate") {
				string mode;
				cin >> mode;
				if (mode == "block") {
					bot.inappropriateMode = BlockInappropriate;
				} else if (mode == "allow") {
					bot.inappropriateMode = AllowInappropriate;
				} else if (mode == "boost") {
					bot.inappropriateMode = BoostInappropriate;
				} else {
					fail(
						"Inappropriate inappropriate mode. Expected one of [block, allow, boost].");
					abort();
				}
				continue;
			}
			CardType type2;
			if (type == string(1, color))
				type2 = CardType::MINE;
			else if (type == "b" || type == "r")
				type2 = CardType::OPPONENT;
			else if (type == "c")
				type2 = CardType::CIVILIAN;
			else if (type == "a")
				type2 = CardType::ASSASSIN;
			else {
				fail("Invalid type.");
				abort();
			}

			string word;
			cin >> word;
			if (!bot.engine.wordExists(word)) {
				cout << "{\"status\": 2, \"message\": \"Unknown word: '"
					 << escapeJSON(denormalize(word)) << "'.\"}";
				return;
			}

			bot.addBoardWord(type2, word);
		}

		int firstResult, numResults;
		cin >> firstResult >> numResults;
		if (firstResult < 0)
			fail("Invalid index");
		if (numResults <= 0)
			fail("Invalid count");
		firstResult = min(firstResult, 1000000);
		numResults = min(numResults, 1000000);
		vector<Bot::Result> results = bot.findBestWords(firstResult + numResults);

		if (firstResult >= (int)results.size()) {
			cout << "{\"status\": 3, \"message\": \"No more clues.\"}";
			return;
		}

		auto type2chr = [color](CardType type) -> char {
			switch (type) {
				case CardType::MINE:
					return color;
				case CardType::OPPONENT:
					return (char)(color ^ 'r' ^ 'b');
				case CardType::CIVILIAN:
					return 'c';
				case CardType::ASSASSIN:
					return 'a';
			}
			abort();
		};

		auto printClue = [&](int index) {
			assert(0 <= index && index < (int)results.size());
			string w = results[index].word;
			int count = results[index].number;
			cout << "  {\"word\": \"" << escapeJSON(denormalize(w)) << "\", ";
			cout << "\"count\": " << count << ", \"why\": [";
			bool first = true;
			trav(item, results[index].valuations) {
				cout << (first ? "\n" : ",\n") << "    {"
					 << "\"score\": " << item.score << ", "
					 << "\"word\": \"" << escapeJSON(denormalize(item.word)) << "\", "
					 << "\"type\": \"" << type2chr(item.type) << "\"}";
				first = false;
			}
			cout << "\n  ]}";
		};

		cout << "{\"status\": 1, \"message\": \"Success.\", \"result\": [" << endl;
		bool first = true;
		rep(i, firstResult, min(firstResult + numResults, (int)results.size())) {
			cout << (first ? "\n" : ",\n");
			printClue(i);
			first = false;
		}
		cout << "\n]}";
	} catch (ios::failure e) {
		fail("Incomplete message.");
	}
}

void simMain() {
	string engine = "conceptnet";
	if (engine == "glove")
		engine = "models/glove.840B.330d.bin";
	else if (engine == "conceptnet")
		engine = "models/conceptnet.bin";
	else if (engine == "conceptnet-swe")
		engine = "models/conceptnet-swedish.bin";
	else
		cerr << "Invalid engine parameter.";

	Dictionary dict;
	Word2VecSimilarityEngine word2vecEngine(dict);
	if (!word2vecEngine.load(engine, false))
		cerr << "Unable to load similarity engine.";

	string line;
	while (getline(cin, line)) {
		stringstream ss(line);

		string query;
		ss >> query;

		string s;
		ss >> s;
		assert(s == ":");
		bool skipped = false;
		while (ss >> s) {
			if (s == ":") {
				skipped = true;
			} else {
				cout << query << " " << s << " "
					 << word2vecEngine.similarity(dict.getID(query), dict.getID(s)) << " "
					 << (skipped ? 0 : 1) << endl;
			}
		}
	}
}

void simMain2() {
	string engine = "conceptnet";
	if (engine == "glove")
		engine = "models/glove.840B.330d.bin";
	else if (engine == "conceptnet")
		engine = "models/conceptnet.bin";
	else if (engine == "conceptnet-swe")
		engine = "models/conceptnet-swedish.bin";
	else
		cerr << "Invalid engine parameter.";

	Dictionary dict;
	Word2VecSimilarityEngine word2vecEngine(dict);
	if (!word2vecEngine.load(engine, false))
		cerr << "Unable to load similarity engine.";

	string line;
	while (getline(cin, line)) {
		stringstream ss(line);

		string query;
		ss >> query;

		string s;
		ss >> s;
		assert(s == ":");
		bool skipped = false;
		vector<string> picked;
		vector<string> all;
		while (ss >> s) {
			if (s == ":") {
				skipped = true;
			} else {
				if (!skipped)
					picked.push_back(s);
				all.push_back(s);
			}
		}

		for (int i = 0; i < (int)picked.size(); i++) {
			for (int j = i + 1; j < (int)all.size(); j++) {
				float s1 = word2vecEngine.similarity(dict.getID(query), dict.getID(picked[i]));
				float s2 = word2vecEngine.similarity(dict.getID(query), dict.getID(all[j]));
				cout << min(s1, s2) << " " << max(s1, s2) << " " << (s1 < s2 ? 1 : 0) << endl;
				cout << max(s1, s2) << " " << min(s1, s2) << " " << (s1 < s2 ? 0 : 1) << endl;
			}
		}
	}
}

int sign(int v) {
	return v > 0 ? 1 : (v < 0 ? -1 : 0);
}

/** -1 if the indices are in reverse order, +1 if they are in increasing order, otherwise somewhere
 * in between. */
float kendallRankCoefficient(vector<int> indices) {
	float numerator = 0;
	int differingPairs = 0;
	for (int i = 0; i < (int)indices.size(); i++) {
		for (int j = 0; j < i; j++) {
			int v = sign(indices[i] - indices[j]);
			if (v != 0) {
				numerator += v;
				differingPairs++;
			}
		}
	}
	return differingPairs > 0 ? numerator / differingPairs : 0.0f;
}

struct Feature {
	double conceptnetSimilarity;
	double conceptnetNorm;
	vector<float> conceptnetVector;
	vector<float> clueConceptnetVector;
	double gloveSimilarity;
	double gloveNorm;
	vector<float> gloveVector;
	vector<float> clueGloveVector;

	void writeTo(ofstream &fout) {
		fout << conceptnetSimilarity << " ";
		fout << conceptnetNorm << " ";
		fout << gloveSimilarity << " ";
		fout << gloveNorm << " ";
		for (auto x : conceptnetVector)
			fout << x << " ";
		/*for(auto x : gloveVector)
			fout << x << " ";*/
		for (auto x : clueConceptnetVector)
			fout << x << " ";
		/*for(auto x : clueGloveVector)
			fout << x << " ";*/
	}
};

void extractFeatures(string trainFileName, string testFileName) {
	Dictionary dict;
	Word2VecSimilarityEngine conceptnetEngine(dict);
	if (!conceptnetEngine.load("models/conceptnet.bin", false))
		cerr << "Unable to load similarity engine.";

	Word2VecSimilarityEngine gloveEngine(dict);
	if (!gloveEngine.load("models/glove.840B.330d.bin", false))
		cerr << "Unable to load similarity engine.";

	ofstream trainFile;
	ofstream testFile;
	trainFile.open(trainFileName);
	if (testFileName != "") {
		testFile.open(testFileName);
	}

	string line;
	while (getline(cin, line)) {
		bool trainSet = (rand() % 10 < 8);  // Use 80% of data for training
		if (testFileName == "")
			trainSet = true;
		stringstream ss(line);

		string query;
		ss >> query;
		if (!conceptnetEngine.wordExists(query) || !gloveEngine.wordExists(query))
			continue;

		string s;
		ss >> s;
		assert(s == ":");
		bool skipped = false;
		vector<string> picked;
		vector<Feature> features;
		vector<string> words;
		while (ss >> s) {
			if (s == ":") {
				skipped = true;
			} else if (conceptnetEngine.wordExists(s) && gloveEngine.wordExists(s)) {
				Feature f;
				f.conceptnetSimilarity =
					conceptnetEngine.similarity(dict.getID(query), dict.getID(s));
				f.conceptnetNorm = conceptnetEngine.stat(dict.getID(s));
				f.conceptnetVector = conceptnetEngine.getVector(dict.getID(s));
				f.clueConceptnetVector = conceptnetEngine.getVector(dict.getID(query));
				f.gloveSimilarity = gloveEngine.similarity(dict.getID(query), dict.getID(s));
				f.gloveNorm = gloveEngine.stat(dict.getID(s));
				f.gloveVector = gloveEngine.getVector(dict.getID(s));
				f.clueGloveVector = gloveEngine.getVector(dict.getID(query));
				for (int i = 0; i < (int)words.size(); i++) {
					if (trainSet) {
						features[i].writeTo(trainFile);
						f.writeTo(trainFile);
						trainFile << endl;
					} else {
						features[i].writeTo(testFile);
						f.writeTo(testFile);
						testFile << endl;
					}
				}
				if (!skipped) {
					features.push_back(f);
					words.push_back(s);
				}
			}
		}
	}

	trainFile.close();
	if (testFileName != "") {
		testFile.close();
	}
}

void benchSimilarity() {
	string engine = "conceptnet";
	if (engine == "glove")
		engine = "models/glove.840B.330d.bin";
	else if (engine == "conceptnet")
		engine = "models/conceptnet.bin";
	else if (engine == "conceptnet-swe")
		engine = "models/conceptnet-swedish.bin";
	else
		cerr << "Invalid engine parameter.";

	Dictionary dict;
	auto word2vecEngine = unique_ptr<SimilarityEngine>(new Word2VecSimilarityEngine(dict));
	if (!word2vecEngine->load(engine, false))
		cerr << "Unable to load similarity engine.";

	auto wikisaurus = unique_ptr<SimilarityEngine>(new EdgeListSimilarityEngine(dict));
	if (!wikisaurus->load("wikisaurus_edges.txt", false))
		cerr << "Unable to load wikisaurus similarity engine.";

	auto randSimilarity = unique_ptr<SimilarityEngine>(new RandomSimilarityEngine());

	MixingSimilarityEngine similarityEngine;
	similarityEngine.engine1 = move(word2vecEngine);
	similarityEngine.multiplier1 = 1;
	//similarityEngine.engine2 = move(randSimilarity);
	similarityEngine.engine2 = move(wikisaurus);
	similarityEngine.multiplier2 = 10;
	//cin >> similarityEngine.multiplier2;

	float sumScore = 0;
	float weight = 0;

	string line;
	while (getline(cin, line)) {
		stringstream ss(line);

		string query;
		ss >> query;
		if (!similarityEngine.wordExists(query))
			continue;

		string s;
		ss >> s;
		assert(s == ":");
		bool skipped = false;
		vector<string> picked;
		vector<pair<float, int>> all;
		vector<string> words;
		int nextIndex = 0;
		while (ss >> s) {
			if (s == ":") {
				skipped = true;
			} else if (similarityEngine.wordExists(s)) {
				float sim = similarityEngine.similarity(dict.getID(query), dict.getID(s));
				all.push_back(make_pair(sim, nextIndex));
				words.push_back(s);
				if (!skipped) {
					nextIndex++;
				}
			}
		}
		sort(all.rbegin(), all.rend());
		vector<int> allIndex;
		for (auto pair : all) {
			allIndex.push_back(pair.second);
		}

		float score = kendallRankCoefficient(allIndex);

		sumScore += score;
		weight += 1;

#if DEBUG_SCORES
		cout << score << endl;
		if (score > 0.5f) {
			cout << query << endl;
			for (auto w : words) {
				float sim = word2vecEngine.similarity(dict.getID(query), dict.getID(w));
				cout << " " << w << " " << sim << endl;
			}
			cout << line << endl;
		}
#endif
	}

	float totalScore = sumScore / weight;
	cout << totalScore << endl;
}

void serverMain() {
	cin.exceptions(ios::failbit | ios::eofbit | ios::badbit);
	srand(time(0));
	stringstream fileName;
	auto t = time(nullptr);
	auto tm = *localtime(&t);
	fileName << "test_data/ordered5/output-" << put_time(&tm, "%Y-%m-%d %H:%M:%S") << ".txt";
	ofstream output(fileName.str());
	if (!output) {
		cerr << "Failed to open output file '" << fileName.str() << "'" << endl;
		return;
	}

	string engine = "conceptnet";
	if (engine == "glove") {
		engine = "models/glove.840B.330d.bin";
	} else if (engine == "conceptnet") {
		engine = "models/conceptnet.bin";
	} else if (engine == "conceptnet-swe") {
		engine = "models/conceptnet-swedish.bin";
	} else {
		cerr << "Invalid engine parameter.";
		return;
	}

	Dictionary dict;
	Word2VecSimilarityEngine word2vecEngine(dict);
	if (!word2vecEngine.load(engine, false))
		cerr << "Unable to load similarity engine.";

	InappropriateEngine inappropriateEngine("inappropriate.txt", dict);

	auto words = dict.getCommonWords(5000);
	// Remove the most common words
	words.erase(words.begin(), words.begin() + 100);

	string COLOR_RED = "\033[31m";
	string COLOR_GREEN = "\033[32m";
	string COLOR_BLUE = "\033[34m";
	string RESET = "\033[0m";

	cout << "Starting" << endl;
	auto wordListFile = ifstream("wordlist-eng.txt");
	vector<string> wordList;
	string wordListWord;
	while (wordListFile >> wordListWord) {
		string normalized = normalize(wordListWord);
		if (word2vecEngine.wordExists(normalized)) {
			wordList.push_back(normalized);
		}
	}
	cout << "Loaded Code Names word list with " << wordList.size() << " words" << endl;

	while (true) {
		string query = dict.getWord(words[rand() % words.size()]);
		vector<string> subset;
		for (int i = 0; i < 5; i++) {
			// Range from -0.2 to 1.2
			float targetSimilarity = rand() / (float)RAND_MAX;
			targetSimilarity = -0.2f + targetSimilarity * 1.4;

			float bestSimilarity = -10;
			string bestWord = "";
			for (auto w : wordList) {
				float similarity = word2vecEngine.similarity(dict.getID(query), dict.getID(w));
				float d1 = abs(similarity - targetSimilarity);
				float d2 = abs(bestSimilarity - targetSimilarity);
				if (d1 < d2 && fmod(rand(), d1 + d2) < d2) {
					if (!superOrSubstring(w, query) &&
						find(subset.begin(), subset.end(), w) == subset.end()) {
						bestSimilarity = similarity;
						bestWord = w;
					}
				}
			}

			subset.push_back(bestWord);
		}

		cout << COLOR_GREEN << query << RESET << endl;
		random_shuffle(subset.begin(), subset.end());
		for (int i = 0; i < (int)subset.size(); i++) {
			cout << COLOR_GREEN << (i + 1) << RESET << ": " << subset[i];
			// cout << " " << word2vecEngine.similarity(dict.getID(query),
			// dict.getID(subset[i]));
			cout << endl;
		}

		bool worked = false;
		while (!worked) {
			worked = true;

			cout << "Enter order as e.g '2 4'. Items which are not included are assumed to be very "
					"unrelated to the query."
				 << endl;
			string line;
			getline(cin, line);

			if (line == "exit" || line == "quit" || line == ":q" || line == ":x") {
				output.close();
				exit(0);
			}

			if (line == "") {
				cout << COLOR_RED << "Are you sure that no words are related to the query? [yes/no]"
					 << RESET << endl;
				string answer;
				getline(cin, answer);
				if (answer != "yes") {
					worked = false;
					continue;
				}
			}

			stringstream ss(line);
			int itemIndex;
			vector<string> picked;
			while (ss >> itemIndex) {
				if (itemIndex < 1 || itemIndex > (int)subset.size()) {
					cout << COLOR_RED << "Index out of range" << RESET << endl;
					worked = false;
					break;
				}

				if (find(picked.begin(), picked.end(), subset[itemIndex - 1]) != picked.end()) {
					cout << COLOR_RED << "Duplicate index" << RESET << endl;
					worked = false;
					break;
				}

				picked.push_back(subset[itemIndex - 1]);
			}

			if (worked) {
				output << query << "\t:\t";
				for (auto w : picked)
					output << w << "\t";
				output << "\t:\t";
				for (auto w : subset) {
					if (find(picked.begin(), picked.end(), w) == picked.end()) {
						output << w << "\t";
					}
				}

				output << "\n";
				output.flush();
				break;
			}
		}
	}
}

int main(int argc, char **argv) {
	if (argc == 2 && argv[1] == string("--server")) {
		serverMain();
		return 0;
	}

	if (argc == 2 && argv[1] == string("--batch")) {
		batchMain();
		return 0;
	}

	if (argc == 2 && argv[1] == string("--bench-similarity")) {
		benchSimilarity();
		return 0;
	}

	if (argc >= 3 && argv[1] == string("--extract-features")) {
		string trainingFile = argv[2];
		string testFile = argc >= 4 ? argv[3] : "";
		if (argc >= 5) {
			srand(atoi(argv[4]));
		}
		extractFeatures(trainingFile, testFile);
		return 0;
	}

	Dictionary dict;
	Word2VecSimilarityEngine word2vecEngine(dict);
	if (!word2vecEngine.load("data.bin", true)) {
		cerr << "Failed to load data.bin" << endl;
		return 1;
	}

	InappropriateEngine inappropriateEngine("inappropriate.txt", dict);

	GameInterface interface(dict, word2vecEngine, inappropriateEngine);
	interface.run();
	return 0;
}
