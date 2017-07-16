#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>
using namespace std;

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define rrep(i, a, b) for (int i = (a)-1; i >= int(b); --i)
#define trav(x, v) for (auto &x : v)
#define all(v) (v).begin(), (v).end()

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

string normalize(string s) {
	trav(c, s) {
		if ('A' <= c && c <= 'Z') {
			c = (char)tolower(c);
		} else if (c == ' ') {
			c = '_';
		}
	}
	return s;
}

string denormalize(string s) {
	trav(c, s) {
		if (c == '_') {
			c = ' ';
		}
	}
	return s;
}

float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

struct SimilarityEngine {
	virtual bool load(const string &fileName, bool verbose) = 0;
	virtual float similarity(wordID fixedWord, wordID dynWord) = 0;
	virtual int getPopularity(wordID id) = 0;
	virtual wordID getID(const string &s) = 0;
	virtual const string &getWord(wordID id) = 0;
	virtual bool wordExists(const string &word) = 0;
	virtual float stat(wordID s) = 0;
	virtual vector<wordID> getCommonWords(int vocabularySize) = 0;
	virtual ~SimilarityEngine() {}
};

struct Word2VecSimilarityEngine : SimilarityEngine {
   private:
	int formatVersion, modelid;
	map<string, wordID> word2id;
	vector<vector<float>> words;
	vector<string> wordsStrings;
	enum Models { GLOVE = 1, CONCEPTNET = 2 };

	// All word vectors are stored normalized -- wordNorms holds their original squared norms.
	// In some embeddings, words that have more (specific) meanings have higher norms.
	vector<float> wordNorms;

	/** Similarity between two word vectors.
	 * Implemented as an inner product. This is the main bottleneck of the
	 * engine, and it gains a lot from being compiled with "-O3 -mavx".
	 */
	float similarity(const vector<float> &v1, const vector<float> &v2) {
		float ret = 0;
		int dim = (int)v1.size();
		rep(i, 0, dim) {
			ret += v1[i] * v2[i];
		}
		return ret;
	}

	/** Arbitrary statistic, in this case the word norm. */
	float stat(wordID s) {
		return wordNorms[s];
	}

   public:
	/** Returns true if successful */
	bool load(const string &fileName, bool verbose) {
		int dimension, numberOfWords;
		modelid = formatVersion = 0;
		ifstream fin(fileName, ios::binary);
		fin.read((char *)&numberOfWords, sizeof numberOfWords);
		if (fin && numberOfWords == -1) {
			fin.read((char *)&formatVersion, sizeof formatVersion);
			fin.read((char *)&modelid, sizeof modelid);
			fin.read((char *)&numberOfWords, sizeof numberOfWords);
		}
		fin.read((char *)&dimension, sizeof dimension);
		if (!fin) {
			cerr << "Failed to load " << fileName << endl;
			return false;
		}
		if (verbose) {
			cerr << "Loading word2vec (" << numberOfWords << " words, " << dimension
				 << " dimensions, model " << modelid << '.' << formatVersion << ")... " << flush;
		}

		const int bufSize = 1 << 16;
		float norm;
		char buf[bufSize];
		string word;
		vector<float> values(dimension);
		vector<float> valuesd;
		words.resize(numberOfWords);
		wordsStrings.resize(numberOfWords);
		wordNorms.resize(numberOfWords);
		rep(i, 0, numberOfWords) {
			int len;
			fin.read((char *)&len, sizeof len);
			if (!fin) {
				cerr << "failed at reading entry " << i << endl;
				return false;
			}
			if (len > bufSize || len <= 0) {
				cerr << "invalid length " << len << endl;
				return false;
			}
			fin.read(buf, len);
			if (formatVersion >= 1) {
				fin.read((char *)&norm, sizeof norm);
			} else {
				norm = 1.0f;
			}
			fin.read((char *)values.data(), dimension * sizeof(float));
			if (!fin) {
				cerr << "failed at reading entry " << i << endl;
				return false;
			}
			word.assign(buf, buf + len);
			valuesd.assign(all(values));
			words[i] = move(valuesd);
			wordsStrings[i] = word;
			word2id[word] = wordID(i);
			wordNorms[i] = norm;
			if (modelid == Models::GLOVE) {
				wordNorms[i] = min(pow(wordNorms[i], 0.4f), 5.3f);
			}
		}
		if (verbose) {
			cerr << "done!" << endl;
		}
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

	float similarity(wordID fixedWord, wordID dynWord) {
		float sim = similarity(words[fixedWord], words[dynWord]);
		if (modelid == Models::GLOVE) {
			return sim * wordNorms[dynWord] / 4.5f;
		} else if (modelid == Models::CONCEPTNET) {
			return (sim <= 0 ? sim : pow(sim, 0.8f) * 1.6f);
		} else {
			return sim;
		}
	}

	/** ID representing a particular word */
	wordID getID(const string &s) {
		return word2id.at(s);
	}

	/** Popularity of a word, the most popular word has a popularity of 1, the second most popular
	 * has a popularity of 2 etc. */
	int getPopularity(wordID id) {
		// Word IDs are the indices of words in the input file, which is assumed to be ordered
		// according to popularity
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
			cout << denormalize(s) << " does not occur in the corpus" << endl;
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

struct Bot {
	enum class CardType { MINE, OPPONENT, CIVILIAN, ASSASSIN };
	enum class Difficulty { EASY, MEDIUM, HARD };

	// Give a similarity bonus to "bad" words
	float marginCivilians;
	float marginOpponentWords;
	float marginAssassins;
	float marginOldClue;

	// Constants used in scoring function based
	// on the sigmoid function of the similarities
	float fuzzyWeightAssassin;
	float fuzzyWeightOpponent;
	float fuzzyWeightMy;
	float fuzzyWeightCivilian;
	float fuzzyWeightOldClue;
	float fuzzyExponent;
	float fuzzyOffset;

	// Assume that we will never succeed if the similarity
	// is at most minSimilarity
	float minSimilarity;

	// Good words with smaller similarity than civilians and opponent
	// spies are worth less
	float multiplierAfterBadWord;

	// How bad is it if there is an opponent word with high similarity
	float weightOpponent;

	// How bad is it if there is a civilian word with high similarity
	float weightCivilian;

	// How important is it that the last good word has greater
	// similarity than the next bad word
	float marginWeight;

	// Number of words that are considered common
	int commonWordLimit;

	// Avoid common words
	float commonWordWeight;

	// Number of words that are not considered rare
	int rareWordLimit;

	// Avoid rare words
	float rareWordWeight;

	// Consider only the 50000 most common words
	int vocabularySize;

	// An approximation of the number of correct words we expect each turn
	float valueOfOneTurn;

	float overlapPenalty;

	// Apply penalties to clues with small numbers based on the number of
	// remaining opponent words
	float desperationFactor[4];

	// Apply a penalty to words that only cover a single word
	float singleWordPenalty;

	// A set of strings for which the bot has already provided clues
	set<string> hasInfoAbout;

	// A list of all clues that have already been given to the team
	vector<wordID> oldClues;

	SimilarityEngine &engine;
	
	void setDifficulty(Difficulty difficulty) {
		if (difficulty == Difficulty::EASY) {
			marginCivilians = 0.07f;
			marginOpponentWords = 0.1f;
			marginAssassins = 0.15f;
			marginOldClue = -0.25f;
			fuzzyWeightAssassin = -0.4f;
			fuzzyWeightOpponent = -0.2f;
			fuzzyWeightMy = 0.1f;
			fuzzyWeightCivilian = -0.1f;
			fuzzyWeightOldClue = -2.0f;
			fuzzyExponent = 15;
			fuzzyOffset = 0.35f;
			minSimilarity = 0.3f;
			multiplierAfterBadWord = 0.5f;
			weightOpponent = -2.0f;
			weightCivilian = -0.4f;
			marginWeight = 0.2f;
			commonWordLimit = 1000;
			commonWordWeight = 0.9f;
			rareWordLimit = 10000;
			rareWordWeight = 0.8f;
			vocabularySize = 30000;
			valueOfOneTurn = 2.0f;
			overlapPenalty = 0.3f;
			desperationFactor[0] = 1.0f;
			desperationFactor[1] = 0.5f;
			desperationFactor[2] = 0.7f;
			desperationFactor[3] = 0.9f;
			singleWordPenalty = -0.5f;
		}
		else if (difficulty == Difficulty::MEDIUM) {
			marginCivilians = 0.02f;
			marginOpponentWords = 0.04f;
			marginAssassins = 0.07f;
			marginOldClue = -0.25f;
			fuzzyWeightAssassin = -0.2f;
			fuzzyWeightOpponent = -0.1f;
			fuzzyWeightMy = 0.1f;
			fuzzyWeightCivilian = -0.05f;
			fuzzyWeightOldClue = -2.0f;
			fuzzyExponent = 15;
			fuzzyOffset = 0.3f;
			minSimilarity = 0.2f;
			multiplierAfterBadWord = 0.7f;
			weightOpponent = -1.5f;
			weightCivilian = -0.2f;
			marginWeight = 0.1f;
			commonWordLimit = 1000;
			commonWordWeight = 0.9f;
			rareWordLimit = 15000;
			rareWordWeight = 0.8f;
			vocabularySize = 50000;
			valueOfOneTurn = 2.5f;
			overlapPenalty = 0.4f;
			desperationFactor[0] = 1.0f;
			desperationFactor[1] = 0.2f;
			desperationFactor[2] = 0.4f;
			desperationFactor[3] = 0.7f;
			singleWordPenalty = -0.5f;
		}
		else {
			assert(difficulty == Difficulty::HARD);
			marginCivilians = 0.01f;
			marginOpponentWords = 0.02f;
			marginAssassins = 0.04f;
			marginOldClue = -0.2f;
			fuzzyWeightAssassin = -0.15f;
			fuzzyWeightOpponent = -0.05f;
			fuzzyWeightMy = 0.1f;
			fuzzyWeightCivilian = -0.03f;
			fuzzyWeightOldClue = -1.5f;
			fuzzyExponent = 15;
			fuzzyOffset = 0.25f;
			minSimilarity = 0.15f;
			multiplierAfterBadWord = 0.8f;
			weightOpponent = -1.2f;
			weightCivilian = -0.15f;
			marginWeight = 0.1f;
			commonWordLimit = 1000;
			commonWordWeight = 0.9f;
			rareWordLimit = 20000;
			rareWordWeight = 0.9f;
			vocabularySize = 50000;
			valueOfOneTurn = 3.0f;
			overlapPenalty = 0.5f;
			desperationFactor[0] = 1.0f;
			desperationFactor[1] = 0.1f;
			desperationFactor[2] = 0.3f;
			desperationFactor[3] = 0.5f;
			singleWordPenalty = -0.5f;
		}
	}


	Bot(SimilarityEngine &engine) : engine(engine) {
		setDifficulty(Difficulty::EASY);
	}

	vector<string> myWords, opponentWords, civilianWords, assassinWords;
	struct BoardWord {
		CardType type;
		string word;
		wordID id;
	};
	vector<BoardWord> boardWords;
	void addBoardWord(CardType type, const string &word) {
		boardWords.push_back({type, word, engine.getID(word)});
	}

	/** True if a is a super or substring of b or vice versa */
	bool superOrSubstring(const string &a, const string &b) {
		auto lowerA = normalize(a);
		auto lowerB = normalize(b);
		return lowerA.find(lowerB) != string::npos || lowerB.find(lowerA) != string::npos;
	}

	bool forbiddenWord(const string &word) {
		for (const BoardWord &w : boardWords) {
			if (superOrSubstring(w.word, word))
				return true;
		}
		return false;
	}

	struct ValuationItem {
		float score;
		string word;
		CardType type;
	};

	pair<float, vector<wordID>> getWordScore(wordID word, vector<ValuationItem> *valuation, bool doInflate) {
		typedef pair<float, BoardWord *> Pa;
		static vector<Pa> v;
		int myWordsLeft = 0, opponentWordsLeft = 0;
		v.clear();
		rep(i, 0, boardWords.size()) {
			float sim = engine.similarity(boardWords[i].id, word);
			if (boardWords[i].type == CardType::CIVILIAN){
				if(doInflate){
					sim += marginCivilians;
				}
			}
			else if (boardWords[i].type == CardType::OPPONENT){
				if(doInflate){
					sim += marginOpponentWords;
				}
				opponentWordsLeft++;
			}
			else if (boardWords[i].type == CardType::ASSASSIN){
				if(doInflate){
					sim += marginAssassins;
				}
			}
			else{
				myWordsLeft++;
			}
			v.push_back({-sim, &boardWords[i]});
		}
		sort(all(v), [&](const Pa &a, const Pa &b) { return a.first < b.first; });

		if (valuation != nullptr) {
			valuation->clear();
			trav(it, v) {
				valuation->push_back({-it.first, it.second->word, it.second->type});
			}
		}

		// Compute a fuzzy score
		float baseScore = 0;
		rep(i, 0, v.size()) {
			float weight;
			switch (v[i].second->type) {
				case CardType::MINE:
					weight = fuzzyWeightMy;
					break;
				case CardType::OPPONENT:
					weight = fuzzyWeightOpponent;
					break;
				case CardType::CIVILIAN:
					weight = fuzzyWeightCivilian;
					break;
				case CardType::ASSASSIN:
					weight = fuzzyWeightAssassin;
					break;
				default:
					abort();
			}
			float contribution = weight * sigmoid((-v[i].first - fuzzyOffset) * fuzzyExponent);
			baseScore += contribution;
		}

		for (auto oldClue : oldClues) {
			float sim = engine.similarity(oldClue, word);
			sim += marginOldClue;
			float contribution = fuzzyWeightOldClue * sigmoid((sim - fuzzyOffset) * fuzzyExponent);
			baseScore += contribution;
		}

		int bestCount = 1;
		float curScore = 0, bestScore = baseScore - 10, lastGood = 0;
		int curCount = 0;
		float mult = 1;
		vector<wordID> targetWords;
		rep(i, 0, v.size()) {
			CardType type = v[i].second->type;
			if (type == CardType::MINE) {
				targetWords.push_back(v[i].second->id);
			}
		}
		rep(i, 0, v.size()) {
			if (-v[i].first < minSimilarity)
				break;
			CardType type = v[i].second->type;
			if (type == CardType::ASSASSIN)
				break;
			if (type == CardType::OPPONENT) {
				curScore += weightOpponent;
				mult *= multiplierAfterBadWord;
				continue;
			}
			if (type == CardType::MINE) {
				lastGood = -v[i].first;
				curScore += mult * sigmoid((-v[i].first - fuzzyOffset) * fuzzyExponent);
				++curCount;
			}
			if (type == CardType::CIVILIAN) {
				curScore += mult * weightCivilian;
				mult *= multiplierAfterBadWord;
				continue;
			}
			float tmpScore = -1;
			rep(j, i + 1, v.size()) {
				CardType type2 = v[j].second->type;
				if (type2 == CardType::ASSASSIN || type2 == CardType::OPPONENT) {
					tmpScore =
						mult * marginWeight * sigmoid((lastGood - (-v[j].first)) * fuzzyExponent);
					break;
				}
			}
			tmpScore += baseScore + curScore;
			if (curCount == 1) {
				tmpScore += singleWordPenalty;
			}
			if (curCount < myWordsLeft - 1 && opponentWordsLeft <= 3) {
				// Apply penalty because we can't win this turn and the opponent will probably win next turn
				tmpScore *= desperationFactor[opponentWordsLeft];
			}
			if (tmpScore > bestScore) {
				bestScore = tmpScore;
				bestCount = curCount;
			}
		}
		while (targetWords.size() > bestCount)
			targetWords.pop_back();

		int popularity = engine.getPopularity(word);
		if (popularity < commonWordLimit)
			bestScore *= commonWordWeight;
		else if (popularity > rareWordLimit)
			bestScore *= rareWordWeight;
		return make_pair(bestScore, targetWords);
	}

	void setWords(const vector<string> &_myWords, const vector<string> &_opponentWords,
				  const vector<string> &_civilianWords, const vector<string> &_assassinWords) {
		myWords = _myWords;
		opponentWords = _opponentWords;
		civilianWords = _civilianWords;
		assassinWords = _assassinWords;
		createBoardWords();
	}

	void createBoardWords() {
		boardWords.clear();
		trav(w, myWords) addBoardWord(CardType::MINE, w);
		trav(w, opponentWords) addBoardWord(CardType::OPPONENT, w);
		trav(w, civilianWords) addBoardWord(CardType::CIVILIAN, w);
		trav(w, assassinWords) addBoardWord(CardType::ASSASSIN, w);
	}

	struct Result {
		string word;
		int number;
		float score;
		vector<ValuationItem> valuations;

		bool operator<(const Result &other) const {
			return score > other.score;
		}
	};

	vector<Result> findBestWords(int count = 20) {
		vector<wordID> candidates = engine.getCommonWords(vocabularySize);
		priority_queue<pair<pair<float, int>, wordID>> pq;
		map<int, int> bitRepresentation;
		int myWordsFound = 0;
		rep(i, 0, boardWords.size()) {
			if (boardWords[i].type == CardType::MINE &&
				!hasInfoAbout.count(engine.getWord(boardWords[i].id))) {
				bitRepresentation[boardWords[i].id] = (1 << myWordsFound);
				++myWordsFound;
			} else {
				bitRepresentation[boardWords[i].id] = 0;
			}
		}

		bool usePlanning = (myWordsFound && myWordsFound <= 9);
		vector<int> minMovesNeeded;
		vector<float> bestScore;
		vector<float> bestClueScore;
		vector<wordID> bestWord;
		vector<int> bestParent;
		if (usePlanning) {
			minMovesNeeded = vector<int>((1 << myWordsFound), 1000);
			bestScore = vector<float>((1 << myWordsFound), -1000);
			bestClueScore = vector<float>((1 << myWordsFound), -1000);
			bestWord = vector<wordID>(1 << myWordsFound);
			bestParent = vector<int>(1 << myWordsFound);
			minMovesNeeded[0] = 0;
			bestScore[0] = 0;
		}

		for (wordID candidate : candidates) {
			pair<float, vector<wordID>> res = getWordScore(candidate, nullptr, true);
			pq.push({{res.first, -((int)res.second.size())}, candidate});
			if (res.second.size() > 0 && usePlanning) {
				int bits = 0;
				for (int matchedWord : res.second) {
					bits |= bitRepresentation[matchedWord];
				}
				float newScore = res.first - valueOfOneTurn;
				if (bits && newScore > bestScore[bits] &&
					!forbiddenWord(engine.getWord(candidate))) {
					minMovesNeeded[bits] = 1;
					bestScore[bits] = newScore;
					bestWord[bits] = candidate;
					bestParent[bits] = 0;
				}
			}
		}

		vector<Result> res;

		if (usePlanning) {
			// Make a plan so that every word is covered by a clue in as few moves as possible

			for (int i = 0; i < (1 << myWordsFound); ++i) {
				if (minMovesNeeded[i] != 1)
					continue;
				for (int j = 0; j < (1 << myWordsFound); ++j) {
					int k = (i | j);
					if (k == i || k == j)
						continue;
					float newScore = bestScore[i] + bestScore[j];
					newScore -= __builtin_popcount(i & j) * overlapPenalty;
					if (newScore > bestScore[k]) {
						minMovesNeeded[k] = minMovesNeeded[j] + 1;
						bestScore[k] = newScore;
						bestClueScore[k] = bestScore[i];
						bestParent[k] = j;
						bestWord[k] = bestWord[i];
					}
				}
			}

			// Reconstruct the best sequence of words
			int bits = (1 << myWordsFound) - 1;
			while (bits != 0) {
				wordID word = bestWord[bits];
				bits = bestParent[bits];
				vector<ValuationItem> val;
				auto wordScore = getWordScore(word, &val, false);
				float score = wordScore.first;
				int number = (int)wordScore.second.size();
				res.push_back(Result{engine.getWord(word), number, score, val});
			}
			sort(all(res));
			return res;
		}

		// Extract the top 'count' words that are not forbidden by the rules
		while ((int)res.size() < count && !pq.empty()) {
			auto pa = pq.top();
			pq.pop();
			if (!forbiddenWord(engine.getWord(pa.second))) {
				float score = pa.first.first;
				int number = -pa.first.second;
				wordID word = pa.second;
				vector<ValuationItem> val;
				getWordScore(word, &val, false);
				res.push_back(Result{engine.getWord(word), number, score, val});
			}
		}

		return res;
	}

	void setHasInfo(string word) {
		hasInfoAbout.insert(word);
	}

	void addOldClue(string clue) {
		if (engine.wordExists(clue)) {
			auto wordID = engine.getID(clue);
			oldClues.push_back(wordID);
		}
	}
};

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

class GameInterface {
	typedef Bot::ValuationItem ValuationItem;
	typedef Bot::Result Result;
	typedef Bot::CardType CardType;
	SimilarityEngine &engine;
	Bot bot;
	vector<string> myWords, opponentWords, civilianWords, assassinWords;
	string myColor;

	void printValuation(const string &word, const vector<Bot::ValuationItem> &valuation) {
		cout << "Printing statistics for \"" << denormalize(word) << "\"" << endl;
		map<CardType, string> desc;
		desc[CardType::MINE] = "(My)";
		desc[CardType::OPPONENT] = "(Opponent)";
		desc[CardType::CIVILIAN] = "(Civilian)";
		desc[CardType::ASSASSIN] = "(Assassin)";
		trav(item, valuation) {
			cout << setprecision(6) << fixed << item.score << "\t";
			cout << denormalize(item.word) << " " << desc[item.type] << endl;
		}
		cout << endl;
	}

	void commandReset() {
		myWords.clear();
		opponentWords.clear();
		civilianWords.clear();
		assassinWords.clear();
		bot.setWords(myWords, opponentWords, civilianWords, assassinWords);
	}

	void commandSuggestWord() {
		cout << "Thinking..." << endl;
		vector<Result> results = bot.findBestWords();
		if (results.empty()) {
			cout << "Not a clue." << endl;
		} else {
			Result &best = results[0];
			printValuation(best.word, best.valuations);

			// Print a list with the best clues
			rep(i, 0, (int)results.size()) {
				auto res = results[i];
				cout << (i + 1) << "\t" << setprecision(3) << fixed << res.score << "\t"
					 << engine.stat(engine.getID(res.word)) << "\t" << res.word << " " << res.number
					 << endl;
			}
			cout << endl;

			int p = engine.getPopularity(engine.getID(best.word));
			cout << "The best clue found is " << denormalize(best.word) << " " << best.number
				 << endl;
			cout << best.word << " is the " << p << orderSuffix(p) << " most popular word" << endl;
		}
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
			cout << " " << denormalize(word);
		}
		cout << endl;
		cout << "Opponent spies:";
		for (auto word : opponentWords) {
			cout << " " << denormalize(word);
		}
		cout << endl;
		cout << "Civilians:";
		for (auto word : civilianWords) {
			cout << " " << denormalize(word);
		}
		cout << endl;
		cout << "Assassins:";
		for (auto word : assassinWords) {
			cout << " " << denormalize(word);
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
			v = &civilianWords;
		} else if (command == "a") {
			v = &assassinWords;
		} else if (command == "-") {
			string word;
			cin >> word;
			word = normalize(word);
			eraseFromVector(word, myWords);
			eraseFromVector(word, opponentWords);
			eraseFromVector(word, civilianWords);
			eraseFromVector(word, assassinWords);
		}

		if (v != NULL) {
			string word;
			cin >> word;
			word = normalize(word);
			if (engine.wordExists(word)) {
				v->push_back(word);
			} else {
				cout << denormalize(word) << " was not found in the dictionary" << endl;
			}
		}
		bot.setWords(myWords, opponentWords, civilianWords, assassinWords);
	}

	void commandScore() {
		string word;
		cin >> word;
		if (!engine.wordExists(word)) {
			cout << denormalize(word) << " was not found in the dictionary" << endl;
			return;
		}
		vector<ValuationItem> val;
		pair<float, vector<wordID>> res = bot.getWordScore(engine.getID(word), &val, true);
		printValuation(word, val);
		cout << denormalize(word) << " " << res.second.size() << " has score " << res.first << endl;
	}

	string inputColor() {
		while (true) {
			string color;
			cin >> color;
			color = normalize(color);
			if (color == "b" || color == "blue") {
				return "b";
			}
			if (color == "r" || color == "red") {
				return "r";
			}
		}
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
			command = normalize(command);

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

		Word2VecSimilarityEngine word2vecEngine;
		if (!word2vecEngine.load(engine, false))
			fail("Unable to load similarity engine.");

		Bot bot(word2vecEngine);

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

int main(int argc, char **argv) {
	if (argc == 2 && argv[1] == string("--batch")) {
		batchMain();
		return 0;
	}

	Word2VecSimilarityEngine word2vecEngine;
	if (!word2vecEngine.load("data.bin", true)) {
		cerr << "Failed to load data.bin" << endl;
		return 1;
	}

	GameInterface interface(word2vecEngine);
	interface.run();
	return 0;
}
