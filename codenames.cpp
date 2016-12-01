#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define rrep(i, a, b) for (int i = (a)-1; i >= int(b); --i)
#define trav(x, v) for (auto &x : v)
#define all(v) (v).begin(), (v).end()
#define what_is(x) cout << #x << " is " << x << endl;

typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<pii> vpi;

/** Represents a single word or phrase in a similarity engine */
enum wordID : int {};

void eraseFromVector(string word, vector<string> &v) {
	rep(i, 0, v.size()) {
		if (v[i] == word) {
			v.erase(v.begin() + i);
			i--;
		}
	}
}

string toLowerCase(string s) {
	rep(i, 0, s.size()) {
		if ('A' <= s[i] && s[i] <= 'Z') {
			s[i] = (char)tolower(s[i]);
		}
	}
	return s;
}

float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

struct SimilarityEngine {
	virtual bool load(const char *fileName) = 0;
	virtual float similarity(wordID s1, wordID s2) = 0;
	virtual int getPopularity(wordID id) = 0;
	virtual wordID getID(const string &s) = 0;
	virtual const string &getWord(wordID id) = 0;
	virtual bool wordExists(const string &word) = 0;
	virtual vector<wordID> getCommonWords(int vocabularySize) = 0;
	virtual ~SimilarityEngine() {}
};

struct Word2VecSimilarityEngine : SimilarityEngine {
   private:
	map<string, wordID> word2id;
	vector<vector<float>> words;
	vector<string> wordsStrings;

	/** Similarity between two word vectors.
	 * Implemented as an inner product.
	 */
	float similarity(const vector<float> &v1, const vector<float> &v2) {
		float ret = 0;
		int dim = (int)v1.size();
		rep(i, 0, dim) {
			ret += v1[i] * v2[i];
		}
		return ret;
	}

   public:
	/** Returns true if successful */
	bool load(const char *fileName) {
		int dimension, numberOfWords;
		ifstream fin(fileName, ios::binary);
		fin.read((char *)&numberOfWords, sizeof numberOfWords);
		fin.read((char *)&dimension, sizeof dimension);
		if (!fin) {
			cerr << "Failed to load " << fileName << endl;
			return false;
		}
		cerr << "Loading word2vec (" << numberOfWords << " words, "
			 << dimension << " dimensions)..." << flush;

		const int bufSize = 1 << 16;
		char buf[bufSize];
		string word;
		vector<float> values(dimension);
		vector<float> valuesd;
		words.resize(numberOfWords);
		wordsStrings.resize(numberOfWords);
		rep(i, 0, numberOfWords) {
			int len;
			fin.read((char *)&len, sizeof len);
			if (!fin) {
				cerr << " failed at reading entry " << i << endl;
				return false;
			}
			if (len > bufSize || len <= 0) {
				cerr << " invalid length " << len << endl;
				return false;
			}
			fin.read(buf, len);
			fin.read((char *)values.data(), dimension * sizeof(float));
			if (!fin) {
				cerr << " failed at reading entry " << i << endl;
				return false;
			}
			word.assign(buf, buf + len);
			valuesd.assign(all(values));
			words[i] = move(valuesd);
			wordsStrings[i] = word;
			word2id[word] = wordID(i);
		}
		cerr << " done!" << endl;
		return true;
	}

	/** Top N most popular words */
	vector<wordID> getCommonWords(int vocabularySize) {
		vector<wordID> ret;
		vocabularySize = min(vocabularySize, (int)words.size());
		ret.reserve(vocabularySize);
		for (int i = 0; i < vocabularySize; i++) {
			ret.push_back(wordID(i));
		}
		return ret;
	}

	float similarity(wordID s1, wordID s2) {
		return similarity(words[s1], words[s2]);
	}

	/** ID representing a particular word */
	wordID getID(const string &s) {
		return word2id.at(s);
	}

	/** Popularity of a word, the most popular word has a popularity of 1, the second most popular has a popularity of 2 etc. */
	int getPopularity(wordID id) {
		// Word IDs are the indices of words in the input file, which is assumed to be ordered according to popularity
		return id + 1;
	}

	/** Word string corresponding to the ID */
	const string &getWord(wordID id) {
		return wordsStrings[id];
	}

	/** True if the word2vec model includes a vector for the specified word */
	bool wordExists(const string &word) {
		return word2id.count(word) > 0;
	}

	vector<pair<float, string>> similarWords(const string &s) {
		if (!wordExists(s)) {
			cout << s << " does not occur in the corpus" << endl;
			return vector<pair<float, string>>();
		}
		vector<pair<float, wordID>> ret;
		rep(i, 0, (int)words.size()) {
			ret.push_back(make_pair(-similarity(getID(s), wordID(i)), wordID(i)));
		}
		sort(all(ret));
		vector<pair<float, string>> res;
		rep(i, 0, 10) {
			res.push_back(make_pair(-ret[i].first, getWord(ret[i].second)));
		}
		return res;
	}
};

struct GraphSimilarityEngine {
	class Node {
	public:
		string word;
		vector<pair<float,Node*>> edges;
		int types;

		bool canBeNoun() {
			return (types >> 0) & 1;
		}

		bool canBeVerb() {
			return (types >> 1) & 1;
		}

		bool canBeAdj() {
			return (types >> 2) & 1;
		}
	};

	SimilarityEngine word2vecEngine;
	map<string, Node*> word2node;
	vector<Node*> nodes;

	GraphSimilarityEngine(SimilarityEngine word2vecEngine) : word2vecEngine(word2vecEngine) {}

	// @author Stack Overflow
	void split(const string &s, char delim, vector<string> &elems) {
	    stringstream ss;
	    ss.str(s);
	    string item;
	    while (getline(ss, item, delim)) {
	        elems.push_back(item);
	    }
	}

	// @author Stack Overflow
	// trim from start (in place)
	static inline void ltrim(string &s) {
	    s.erase(s.begin(), find_if(s.begin(), s.end(),
	            not1(ptr_fun<int, int>(isspace))));
	}

	// @author Stack Overflow
	// trim from end (in place)
	static inline void rtrim(string &s) {
	    s.erase(find_if(s.rbegin(), s.rend(),
	            not1(ptr_fun<int, int>(isspace))).base(), s.end());
	}

	// @author Stack Overflow
	// trim from both ends (in place)
	static inline void trim(string &s) {
	    ltrim(s);
	    rtrim(s);
	}

	Node* getOrCreateNode(const string& word) {
		auto it = word2node.find(word);
		if (it == word2node.end()) {
			auto node = new Node();
			node->word = word;
			nodes.push_back(node);
			it = word2node.insert(make_pair(word, node)).first;
		}
		return it->second;
	}

	Node* getNode(string word) {
		if (word2node.find(word) != word2node.end()) {
			return word2node[word];
		} else {
			return nullptr;
		}
	}

	typedef map<Node*, pair<double, Node*>> SearchResult;

	SearchResult search(string root, double maxDistance) {
		auto rootNode = getNode(root);
		map<Node*, pair<double, Node*>> distsAndParents;
		priority_queue<pair<double,Node*>, vector<pair<double,Node*>>, greater<pair<double,Node*>>> que;
		que.push({0.0, rootNode});
		distsAndParents[rootNode] = {0, nullptr};

		if (word2vecEngine.wordExists(root)) {
			for (auto& commonWord : word2vecEngine.getCommonWords(10000)) {
				auto node = getNode(*commonWord);
				if (node != nullptr && node != rootNode) {
					auto dist = word2vecEngine.similarity(root, *commonWord);
					dist = abs(1 - dist);
					auto cost = dist;
					distsAndParents[node] = {cost, rootNode};
					que.push({cost, node});
				}
			}
		}

		while(!que.empty()) {
			auto p = que.top();
			que.pop();
			double dist = p.first;
			auto& dipa = distsAndParents[p.second];
			if (dipa.first < dist) {
				// Already visited
				continue;
			}
			assert(dipa.first == dist);

			for (auto edge : p.second->edges) {
				auto newDist = dist + edge.first;
				if (newDist > maxDistance) continue;
				auto& dipa2 = distsAndParents[edge.second];
				if (dipa2.second == nullptr || newDist < dipa2.first) {
					dipa2 = {newDist, p.second};
					que.push({newDist, edge.second});
				}
			}
		}

		return distsAndParents;
	}

	bool load(const char *fileName, const char *fileName2) {
#ifdef SYNONYM_DATABASE
		ifstream fin(fileName, ios::binary);
		while(true) {
			string word;
			int meanings;
			getline(fin, word, '|');
			if (!fin) {
				cerr << "Done" << endl;
				break;
			}
			trim(word);
			fin >> meanings;

			Node* node = getOrCreateNode(word);
			
			for (int i = 0; i < meanings; i++) {
				string type;
				fin.ignore(255, '(');
				getline(fin, type, ')');
				fin.ignore(255, '|');
				string synonymString;
				getline(fin, synonymString);

				if (type == "noun") {
					node->types |= 1 << 0;
				} else if (type == "verb") {
					node->types |= 1 << 1;
				} else if (type == "adj") {
					node->types |= 1 << 2;
				} else if (type == "adv") {
					// Eh, sort of like adj
					node->types |= 1 << 2;
				} else {
					cerr << "Did not recognise type: " << type << endl;
				}
			}
		}

		// Read again, now with more information
		fin = ifstream(fileName, ios::binary);
		while(true) {
			string word;
			int meanings;
			getline(fin, word, '|');
			if (!fin) {
				cerr << "Done" << endl;
				break;
			}
			trim(word);
			fin >> meanings;

			Node* node = getOrCreateNode(word);
			
			for (int i = 0; i < meanings; i++) {
				string type;
				fin.ignore(255, '(');
				getline(fin, type, ')');
				fin.ignore(255, '|');
				string synonymString;
				getline(fin, synonymString);
				vector<string> synonyms;
				split(synonymString, '|', synonyms);

				for (int j = 0; j < synonyms.size(); j++) {
					bool isCategory = j == 0;
					auto syn = synonyms[j];
					trim(syn);
					if (syn.size() == 0) continue;

					node->edges.push_back({isCategory ? 1.5 : 2, getOrCreateNode(syn)});
					getOrCreateNode(syn)->edges.push_back({isCategory ? 1.5 : 2, node});

					vector<string> subWords;
					split(syn, ' ', subWords);
					if (subWords.size() > 1) {
						int numNonAdj = 0;
						for (auto sub : subWords) {
							trim(sub);
							if (sub.size() <= 2) continue;
							auto subNode = getOrCreateNode(sub);
							if (subNode->canBeVerb() || subNode->canBeNoun()) numNonAdj++;
						}

						if (numNonAdj == 1) {
							for (auto sub : subWords) {
								trim(sub);
								if (sub.size() <= 2) continue;
								auto subNode = getOrCreateNode(sub);
								if (subNode->canBeVerb() || subNode->canBeNoun()) {
									//cerr << "Adding edge " << node->word << " " << subNode->word << endl;
									node->edges.push_back({2.2, subNode});
									subNode->edges.push_back({2.2, node});
								}
							}
						}
					}
				}
				//cerr << type << ": " << synonyms << endl;
			}
			//cerr << word << endl;
		}
#else
		int counter = 0;
		auto fin = ifstream(fileName2, ios::binary);
		cerr << "Loading..." << endl;
		string word1;
		string word2;
		while(true) {
			int weight;
			getline(fin, word1, '\t');
			getline(fin, word2, '\t');
			fin >> weight;
			// Newline
			fin.ignore(1);

			if (!fin) {
				cerr << endl;
				cerr << "Done" << endl;
				cerr << "Read " << counter << " entries with " << nodes.size() << " nodes" << endl;
				break;
			}

			counter++;
			if ((counter % 100000) == 0) {
				cerr << "\rRead " << counter << " entries...";
			}

			Node* node1 = getOrCreateNode(word1);
			Node* node2 = getOrCreateNode(word2);
			
			node2->edges.push_back({ 1.0 / sqrt(weight), node1});
			node1->edges.push_back({ 10.0 / sqrt(weight) + 0.5, node2});
		}
#endif

		return true;
	}
};

struct Bot {
	// Give a similarity bonus to "bad" words
	float marginCivilians = 0.02f;
	float marginOpponentWords = 0.04f;
	float marginAssassins = 0.07f;

	// Constants used in scoring function based
	// on the sigmoid function of the similarities
	float fuzzyWeightAssassin = -0.2f;
	float fuzzyWeightOpponent = -0.1f;
	float fuzzyWeightMy = 0.1f;
	float fuzzyWeightGrey = -0.05f;
	float fuzzyExponent = 15;
	float fuzzyOffset = 0.3f;

	// Assume that we will never succeed if the similarity
	// is at most minSimilarity
	float minSimilarity = 0.2f;

	// Good words with smaller similarity than civilians and opponent
	// spies are worth less
	float multiplierAfterBadWord = 0.7f;

	// How bad is it if there is an opponent word with high similarity
	float weightOpponent = -1.5f;

	// How bad is it if there is a grey word with high similarity
	float weightGrey = -0.2f;

	// How important is it that the last good word has greater
	// similarity than the next bad word
	float marginWeight = 0.1f;

	// Number of words that are considered common
	int commonWordLimit = 1000;

	// Avoid common words
	float commonWordWeight = 0.9f;

	// Number of words that are not considered rare
	int rareWordLimit = 15000;

	// Avoid rare words
	float rareWordWeight = 0.8f;

	// Consider only the 50000 most common words
	int vocabularySize = 50000;

	SimilarityEngine &engine;

	Bot(SimilarityEngine &engine) : engine(engine) {}

	vector<string> myWords, opponentWords, greyWords, assassinWords;
	struct BoardWord {
		char type;
		string word;
		wordID id;
	};
	vector<BoardWord> boardWords;
	void addBoardWord(char type, const string &word) {
		boardWords.push_back({
			type,
			word,
			engine.getID(word)
		});
	}

	/** True if a is a super or substring of b or vice versa */
	bool superOrSubstring(const string &a, const string &b) {
		auto lowerA = toLowerCase(a);
		auto lowerB = toLowerCase(b);
		return lowerA.find(lowerB) != string::npos || lowerB.find(lowerA) != string::npos;
	}

	bool forbiddenWord(const string &word) {
		for (const BoardWord &w : boardWords) {
			if (superOrSubstring(w.word, word))
				return true;
		}
		return false;
	}

	pair<float, int> getWordScore(wordID word, bool debugPrint) {
		if (debugPrint)
			cout << "Printing statistics for \"" << engine.getWord(word) << "\"" << endl;

		typedef pair<float, BoardWord *> Pa;
		static vector<Pa> v;
		v.clear();
		rep(i, 0, boardWords.size()) {
			float sim = engine.similarity(boardWords[i].id, word);
			if (boardWords[i].type == 'g')
				sim += marginCivilians;
			if (boardWords[i].type == 'o')
				sim += marginOpponentWords;
			if (boardWords[i].type == 'a')
				sim += marginAssassins;
			v.push_back({-sim, &boardWords[i]});
		}
		sort(all(v), [&](const Pa &a, const Pa &b) { return a.first < b.first; });

		if (debugPrint) {
			rep(i, 0, v.size()) {
				cout << setprecision(6) << fixed << -v[i].first << "\t" << v[i].second->word << " ";
				switch (v[i].second->type) {
					case 'm':
						cout << "(My)" << endl;
						break;
					case 'o':
						cout << "(Opponent)" << endl;
						break;
					case 'g':
						cout << "(Civilian)" << endl;
						break;
					case 'a':
						cout << "(Assassin)" << endl;
						break;
					default:
						assert(0);
				}
			}
			cout << endl;
		}

		// Compute a fuzzy score
		float baseScore = 0;
		rep(i, 0, v.size()) {
			char type = v[i].second->type;
			float weight;
			if (type == 'a')
				weight = fuzzyWeightAssassin;
			else if (type == 'o')
				weight = fuzzyWeightOpponent;
			else if (type == 'm')
				weight = fuzzyWeightMy;
			else if (type == 'g')
				weight = fuzzyWeightGrey;
			else
				assert(0);
			float contribution = weight * sigmoid((-v[i].first - fuzzyOffset) * fuzzyExponent);
			baseScore += contribution;
		}

		int bestCount = 0;
		float curScore = 0, bestScore = 0, lastGood = 0;
		int curCount = 0;
		float mult = 1;
		rep(i, 0, v.size()) {
			if (-v[i].first < minSimilarity)
				break;
			char type = v[i].second->type;
			if (type == 'a')
				break;
			if (type == 'o') {
				curScore += weightOpponent;
				mult *= multiplierAfterBadWord;
				continue;
			}
			if (type == 'm') {
				lastGood = -v[i].first;
				curScore += mult * sigmoid((-v[i].first - fuzzyOffset) * fuzzyExponent);
				++curCount;
			}
			if (type == 'g') {
				curScore += mult * weightGrey;
				mult *= multiplierAfterBadWord;
				continue;
			}
			float tmpScore = -1;
			rep(j, i + 1, v.size()) {
				char type2 = v[j].second->type;
				if (type2 == 'a' || type2 == 'o') {
					tmpScore = mult * marginWeight * sigmoid((lastGood - (-v[j].first)) * fuzzyExponent);
					break;
				}
			}
			tmpScore += baseScore + curScore;
			if (tmpScore > bestScore) {
				bestScore = tmpScore;
				bestCount = curCount;
			}
		}

		int popularity = engine.getPopularity(word);
		if (popularity < commonWordLimit)
			bestScore *= commonWordWeight;
		else if (popularity > rareWordLimit)
			bestScore *= rareWordWeight;
		return make_pair(bestScore, bestCount);
	}

	pair<float, int> getWordScore(const string &word, bool debugPrint) {
		return getWordScore(engine.getID(word), debugPrint);
	}

	void setWords(const vector<string> &_myWords,
				  const vector<string> &_opponentWords,
				  const vector<string> &_greyWords,
				  const vector<string> &_assassinWords) {
		myWords = _myWords;
		opponentWords = _opponentWords;
		greyWords = _greyWords;
		assassinWords = _assassinWords;
		createBoardWords();
	}

	void createBoardWords() {
		boardWords.clear();
		trav(w, myWords) addBoardWord('m', w);
		trav(w, opponentWords) addBoardWord('o', w);
		trav(w, greyWords) addBoardWord('g', w);
		trav(w, assassinWords) addBoardWord('a', w);
	}

	/** Returns suffix on number such as 'th' for 5 or 'nd' for 2 */
	string orderSuffix(int p) {
		if (p % 10 == 1 && p % 100 != 11) {
			return "st";
		} else if (p % 10 == 2 && p % 100 != 12) {
			return "nd";
		} else if (p % 10 == 3 && p % 100 != 13) {
			return "rd";
		} else {
			return "th";
		}
	}

	pair<string, int> getBestWord() {
		vector<wordID> candidates = engine.getCommonWords(vocabularySize);
		priority_queue<pair<pair<float, int>, wordID>> pq;
		for (wordID candidate : candidates) {
			pair<float, int> res = getWordScore(candidate, false);
			pq.push({{res.first, -res.second}, candidate});
		}

		vector<pair<pair<float, int>, wordID>> topList;

		// Extract the top 20 words that are not forbidden by the rules
		while (topList.size() < 20 && !pq.empty()) {
			auto pa = pq.top();
			pq.pop();
			if (!forbiddenWord(engine.getWord(pa.second)))
				topList.push_back(pa);
		}

		// Print how the score of the best word was computed
		assert(!topList.empty());
		wordID bestWord = topList[0].second;
		int bestCount = -topList[0].first.second;
		getWordScore(bestWord, true);

		// Print a list with the best clues
		rep(i, 0, (int)topList.size()) {
			auto res = topList[i];
			cout << (i + 1) << "\t" << setprecision(3) << fixed << res.first.first << "\t"
				 << engine.getWord(res.second) << " " << -res.first.second << endl;
		}

		int p = engine.getPopularity(bestWord);
		cout << "The best clue found is " << engine.getWord(bestWord) << " " << bestCount << endl;
		cout << engine.getWord(bestWord) << " is the " << p << orderSuffix(p);

		cout << " most popular word" << endl;
		return make_pair(engine.getWord(bestWord), bestCount);
	}
};

class GameInterface {
	SimilarityEngine &engine;
	Bot bot;
	vector<string> myWords, opponentWords, greyWords, assassinWords;
	string myColor;

	void commandReset() {
		myWords.clear();
		opponentWords.clear();
		greyWords.clear();
		assassinWords.clear();
		bot.setWords(myWords, opponentWords, greyWords, assassinWords);
	}

	void commandSuggestWord() {
		cout << "Thinking..." << endl;
		bot.getBestWord();
	}

	void commandHelp() {
		cout << "The following commands are available:" << endl << endl;
		cout << "r <word>\t-\tAdd a red spy to the board" << endl;
		cout << "b <word>\t-\tAdd a blue spy to the board" << endl;
		cout << "c <word>\t-\tAdd a civilian to the board" << endl;
		cout << "a <word>\t-\tAdd an assassin to the board" << endl;
		cout << "- <word>\t-\tRemove a word from the board" << endl;
		cout << "go\t\t-\tReceive clues" << endl;
		cout << "reset\t\t-\tClear the board" << endl;
		cout << "board\t\t-\tPrints the words currently on the board" << endl;
		cout << "score <word>\t-\tCompute how good a given clue would be" << endl;
		cout << "quit\t\t-\tTerminates the program" << endl;
	}

	void commandBoard() {
		cout << "My spies:";
		for (auto word : myWords) {
			cout << " " << word;
		}
		cout << endl;
		cout << "Opponent spies:";
		for (auto word : opponentWords) {
			cout << " " << word;
		}
		cout << endl;
		cout << "Civilians:";
		for (auto word : greyWords) {
			cout << " " << word;
		}
		cout << endl;
		cout << "Assassins:";
		for (auto word : assassinWords) {
			cout << " " << word;
		}
		cout << endl;
	}

	void commandModifyBoard(const string &command) {
		vector<string> *v = NULL;
		if (command == myColor) {
			v = &myWords;
		} else if (command == "b" || command == "r") {
			v = &opponentWords;
		} else if (command == "g" || command == "c") {
			v = &greyWords;
		} else if (command == "a") {
			v = &assassinWords;
		} else if (command == "-") {
			string word;
			cin >> word;
			word = toLowerCase(word);
			eraseFromVector(word, myWords);
			eraseFromVector(word, opponentWords);
			eraseFromVector(word, greyWords);
			eraseFromVector(word, assassinWords);
		}

		if (v != NULL) {
			string word;
			cin >> word;
			word = toLowerCase(word);
			if (engine.wordExists(word)) {
				v->push_back(word);
			} else {
				cout << word << " was not found in the dictionary" << endl;
			}
		}
		bot.setWords(myWords, opponentWords, greyWords, assassinWords);
	}

	void commandScore() {
		string word;
		cin >> word;
		if (!engine.wordExists(word)) {
			cout << word << " was not found in the dictionary" << endl;
			return;
		}
		pair<float, int> res = bot.getWordScore(word, true);
		cout << word << " " << res.second << " has score " << res.first << endl;
	}

	string inputColor() {
		string color;
		cin >> color;
		color = toLowerCase(color);
		while (true) {
			if (color == "b" || color == "blue") {
				color = "b";
				break;
			}
			if (color == "r" || color == "red") {
				color = "r";
				break;
			}
			cin >> color;
			color = toLowerCase(color);
		}
		return color;
	}

   public:
	GameInterface(SimilarityEngine &engine) : engine(engine), bot(engine) {}

	void run() {
		cout << "Type \"help\" for help" << endl;
		cout << "My color (b/r): ";
		myColor = inputColor();

		while (true) {
			string command;
			cin >> command;
			if (!cin)
				break;
			command = toLowerCase(command);

			if (command.size() == 1 && string("rgbac-").find(command) != string::npos) {
				commandModifyBoard(command);
			} else if (command == "play" || command == "go") {
				commandSuggestWord();
			} else if (command == "quit" || command == "exit") {
				break;
			} else if (command == "reset") {
				commandReset();
			} else if (command == "help" || command == "\"help\"") {
				commandHelp();
			} else if (command == "board") {
				commandBoard();
			} else if (command == "score") {
				commandScore();
			} else {
				cout << "Unknown command \"" << command << "\"" << endl;
			}
		}
	}
};

int main() {
	Word2VecSimilarityEngine word2vecEngine;
	if (!word2vecEngine.load("data.bin")) {
		cerr << "Failed to load data.bin" << endl;
		return 1;
	}

	GameInterface interface(word2vecEngine);
	#if true
	SimilarityEngine word2vecEngine;
	word2vecEngine.load("data.bin");

	GraphSimilarityEngine engine(word2vecEngine);
	engine.load("th_en_US_new.dat", "prunned-relations.txt");
	cerr << "Searching..." << endl;
	vector<string> targetWords { "teacher", "mine", "chest", "wind", "bug", "giant", "parachute", "lemon", "cold" };
	//vector<string> targetWords = { "lead", "witch", "brush", "mercury", "march", "bolt", "board", "pilot", "nut" };
	//vector<string> targetWords =  { "casino", "pass", "concert", "board", "queen", "hood", "ground", "cross", "court" };

	/*auto res = engine.search("horse", 20);
	for (auto node : res.dists) {
		if (node.first->word == "car") {
			auto cnode = node.first;
			cerr << res.dists[cnode] << ": ";
			if (res.dists.find(cnode) != res.dists.end()) {
				cerr << "\t";
				while(cnode != nullptr) {
					cerr << " -> " << cnode->word;
					cnode = res.parents[cnode];
				}
				cerr << endl;
			}
		}
	}*/

	vector<GraphSimilarityEngine::SearchResult> results;
	for (auto tgWord : targetWords) {
		cerr << "Searching target word " << tgWord << " (of " << targetWords.size() << ")" << endl;
		results.push_back(engine.search(tgWord, 4.5));
		cerr << "Reached " << results[results.size()-1].size() << " nodes" << endl;
	}

	cerr << "Evaluating..." << endl;

	vector<pair<double, pair<int, GraphSimilarityEngine::Node*>>> scores;
	for (auto root : engine.nodes) {
		int found = 0;
		double sqrDistSum = 0.0;
		vector<pair<double,int>> dists;
		for (int j = 0; j < results.size(); j++) {
			auto it = results[j].find(root);
			if (it != results[j].end()) {
				dists.push_back({it->second.first, j});
			} else {
				dists.push_back({1000.0, j});
			}
		}
		sort(dists.begin(), dists.end());

		double accDist = 0.0;
		for (int j = 0; j < dists.size(); j++) {
			accDist += dists[j].first;
			double avgDist = accDist / j;
			if (j > 1 && avgDist <= 5.0) {
				//cerr << "Found " << root->word << " at " << avgDist << endl;
				scores.push_back({ avgDist, {j, root}});
			}
		}
	}

	sort(scores.begin(), scores.end());
	for (int w = 0; w < min(50, (int)scores.size()); w++) {
		auto& score = scores[w];

		double avgDist = score.first;
		auto root = score.second.second;
		int count = score.second.first;

		cerr << root->word << ": " << count << ": (dist: " << avgDist << ")" << endl;


		int found = 0;
		double sqrDistSum = 0.0;
		vector<pair<double,int>> dists;
		for (int j = 0; j < results.size(); j++) {
			auto it = results[j].find(root);
			if (it != results[j].end()) {
				dists.push_back({it->second.first, j});
			} else {
				dists.push_back({1000.0, j});
			}
		}
		sort(dists.begin(), dists.end());


		for (int k = 0; k <= count; k++) {
			int index = dists[k].second;
			auto& result = results[index];
			auto tg = targetWords[index];
			auto cnode = root;
			if (result.find(cnode) != result.end()) {
				cerr << "\t";
				cerr << result[cnode].first << ": ";
				while(cnode != nullptr) {
					cerr << " -> " << cnode->word;
					cnode = result[cnode].second;
				}
				cerr << endl;
			}
		}
	}
	exit(0);
	#endif
	interface.run();
	return 0;
}
