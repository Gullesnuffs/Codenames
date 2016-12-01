#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <ctime>
#include <set>
#include <string>
#include <vector>
using namespace std;

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define rrep(i, a, b) for (int i = (a)-1; i >= int(b); --i)
#define trav(x, v) for (auto &x : v)
#define all(v) (v).begin(), (v).end()
#define what_is(x) cout << #x << " is " << x << endl;
//#define DEBUG_MCTS

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

float inverseSigmoid(float x) {
	if(x > 0.999999)
		return 10000;
	if(x < 0.000001)
		return -10000;
	return -log(1/x-1);
}

struct SimilarityEngine {
	map<string, vector<float>> vec;
	map<string, int> popularity;
	map<pair<wordID, wordID>, float> cache;
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

struct State{
	vector<string> words;
	vector<float> probPlayer[2];
	vector<float> probNotPlayer[2];
	int turn;
	int clueWordsRemaining[2];

	State(){
		turn = 0;
		clueWordsRemaining[0]=0;
		clueWordsRemaining[1]=0;
	}

	void printWords(){
		rep(i,0,(int)words.size()){
			cout << words[i] << " ";
		}
		cout << endl;
	}
	
	void addWord(string word){
		words.push_back(word);
		probPlayer[0].push_back(0);
		probPlayer[1].push_back(0);
		probNotPlayer[0].push_back(1);
		probNotPlayer[1].push_back(1);
	}

	bool operator<(const State &other) const{
		if(words.size() != other.words.size())
			return words.size() < other.words.size();
		rep(i,0,(int)words.size()){
			if(words[i] != other.words[i])
				return words[i] < other.words[i];
			rep(j,0,2){
				if(probPlayer[j][i] != other.probPlayer[j][i])
					return probPlayer[j][i] < other.probPlayer[j][i];
				if(probNotPlayer[j][i] != other.probNotPlayer[j][i])
					return probNotPlayer[j][i] < other.probNotPlayer[j][i];
			}
		}
		return 0;
	}

	void reset(){
		words.clear();
		probPlayer[0].clear();
		probPlayer[1].clear();
		probNotPlayer[0].clear();
		probNotPlayer[1].clear();
	}

	void acceptClue(pair<string, int> clue, int color, SimilarityEngine &engine){
		clueWordsRemaining[color] += clue.second;
		vector<float> probs;
		double lo=-1.0;
		double hi=1.0;
		rep(iter,0,50){
			probs.clear();
			double mid=(lo+hi)/2;
			float probSum=0;
			rep(i,0,(int)words.size()){
				probs.push_back(sigmoid(30*(engine.similarity(engine.getID(words[i]), engine.getID(clue.first))-mid)));
				probSum += probs.back();
			}
			if(probSum > clue.second)
				lo=mid;
			else
				hi=mid;
		}
		rep(i,0,(int)probs.size()){
			probNotPlayer[color][i] *= 1-probs[i];
			probPlayer[color][i] = 1-probNotPlayer[color][i];
		}
	}

	string getGuess(int color){
		vector<float> probs;
		float probSum=0;
		if(!clueWordsRemaining[color]){
#ifdef DEBUG_MCTS
			cerr << "No clue words remaining" << endl;
#endif
			return "";
		}
#ifdef DEBUG_MCTS
		cerr << "Printing guessing statistics" << endl;
#endif
		rep(i,0,(int)words.size()){
			float score = inverseSigmoid(probPlayer[color][i]);
			score -= inverseSigmoid(0.1+0.4*probPlayer[!color][i]);
			if(score < 0)
				score=0;
#ifdef DEBUG_MCTS
			cerr << words[i] << " " << probPlayer[color][i] << " " << probPlayer[!color][i] << " " << score << endl;
#endif
			probs.push_back(score);
			probSum += probs.back();
		}
		if((rand()%1000)/1000.0 >= probSum)
			return "";
		float r=(((rand()*475345LL+rand())%10000000)/10000000.0)*probSum;
		rep(i,0,(int)words.size()){
			r -= probs[i];
			if(r < 0){
#ifdef DEBUG_MCTS
				cerr << "The score of " << words[i] << " is " << probs[i] << endl;
#endif
				return words[i];
			}
		}
		cerr << "probSum = " << probSum << endl;
		cerr << "r = " << r << endl;
		assert(0);
	}

	void performGuess(string guess, int wordColor, int myColor){
		rep(i,0,(int)words.size()){
			if(words[i] == guess){
				words.erase(words.begin()+i);
				probPlayer[0].erase(probPlayer[0].begin()+i);
				probPlayer[1].erase(probPlayer[1].begin()+i);
				probNotPlayer[0].erase(probNotPlayer[0].begin()+i);
				probNotPlayer[1].erase(probNotPlayer[1].begin()+i);
				--i;
			}
		}
		if(wordColor == myColor){
			clueWordsRemaining[myColor]--;
		}
		assert(words.size() == probPlayer[0].size());
		assert(words.size() == probPlayer[1].size());
	}
};

struct BoardWord {
	char type;
	string word;
	wordID id;

	bool operator==(const BoardWord &other) const{
		return word == other.word;
	}

	bool operator!=(const BoardWord &other) const{
		return !(*this == other);
	}

	bool operator<(const BoardWord &other) const{
		return word < other.word;
	}
};

map<wordID, vector<pair<float, BoardWord *> > > savedSimilarities;

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

	vector<wordID> decentWords;

	SimilarityEngine &engine;

	Bot(SimilarityEngine &engine) : engine(engine) {}

	vector<string> myWords, opponentWords, greyWords, assassinWords;
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
		vector<Pa> v;
		auto it = savedSimilarities.find(word);
		if(it != savedSimilarities.end()){
			vector<Pa>* saved = &it->second;
			v.reserve(boardWords.size());
			int j=0;
			rep(i, 0, (int)boardWords.size()){
				while(*((*saved)[j].second) != boardWords[i]){
					++j;
					if(j == saved->size()){
						cerr << "Failed for " << boardWords[i].word << endl;
						cerr << "Saved vector contains " << saved->size() << " elements" << endl;
						rep(k, 0, (int)saved->size()){
							cerr << "Element " << (k+1) << ": " <<(*saved)[k].second->word << endl;
						}
						cerr << endl;
					}
					assert(j < saved->size());
				}
				v.push_back({(*saved)[j].first, &boardWords[i]});
			}
		}
		else{
			rep(i, 0, (int)boardWords.size()) {
				float sim = engine.similarity(boardWords[i].id, word);
				v.push_back({sim, &boardWords[i]});
			}
			savedSimilarities[word]=v;
		}
		rep(i, 0, (int)boardWords.size()) {
			float sim = engine.similarity(boardWords[i].id, word);
			if (boardWords[i].type == 'g')
				sim += marginCivilians;
			if (boardWords[i].type == 'o')
				sim += marginOpponentWords;
			if (boardWords[i].type == 'a')
				sim += marginAssassins;
			v[i].first = -sim;
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

	vector<pair<string, int> > getBestWords(int numWords, bool printDebug, double probabilityConsider) {
		sort(all(boardWords));
		vector<wordID> candidates;
		if(decentWords.empty())
			candidates = engine.getCommonWords(vocabularySize);
		else
			candidates = decentWords;
		priority_queue<pair<pair<float, int>, wordID>> pq;
		for (wordID candidate : candidates) {
			if((rand()%1000)/1000.0 > probabilityConsider)
				continue;
			pair<float, int> res = getWordScore(candidate, false);
			pq.push({{res.first, -res.second}, candidate});
		}

		vector<pair<pair<float, int>, wordID>> topList;

		// Extract the top numWords words that are not forbidden by the rules
		while (topList.size() < numWords && !pq.empty()) {
			auto pa = pq.top();
			pq.pop();
			if (!forbiddenWord(engine.getWord(pa.second)))
				topList.push_back(pa);
		}

		// Print how the score of the best word was computed
		assert(!topList.empty());
		wordID bestWord = topList[0].second;
		int bestCount = -topList[0].first.second;
		if(printDebug){
			getWordScore(bestWord, true);
		}

		// Print a list with the best clues
		vector<pair<string, int> > bestWords;
		rep(i, 0, (int)topList.size()) {
			auto res = topList[i];
			if(printDebug){
				cout << (i + 1) << "\t" << setprecision(3) << fixed << res.first.first << "\t"
					 << engine.getWord(res.second) << " " << -res.first.second << endl;
			}
			bestWords.emplace_back(engine.getWord(res.second), -res.first.second);
		}

		int p = engine.getPopularity(bestWord);
		if(printDebug){
			cout << "The best clue found is " << engine.getWord(bestWord) << " " << bestCount << endl;
			cout << engine.getWord(bestWord) << " is the " << p << orderSuffix(p);
			cout << " most popular word" << endl;
		}
		return bestWords;
	}

	pair<string, int> getBestWord(bool printDebug, double probabilityConsider){
		vector<pair<string, int> > v = getBestWords(20, printDebug, probabilityConsider);
		return v[0];
	}
	
	pair<string, int> getBestWord(bool printDebug){
		return getBestWord(printDebug, 1.0);
	}

	char getWordColor(string word){
		rep(i,0,(int)boardWords.size()){
			if(boardWords[i].word == word){
				return boardWords[i].type;
			}
		}
		assert(0);
	}

	int simulateGuessing(State* state){
		string guess = state->getGuess(state->turn);
		if(guess == ""){
#ifdef DEBUG_MCTS
			cerr << "Stops guessing" << endl;
#endif
			return -2;
		}
#ifdef DEBUG_MCTS
		cerr << "Guess " << guess << endl;
#endif
		int color = -1;
		if(find(all(myWords), guess) != myWords.end()){
			color=0;
		}
		else if(find(all(opponentWords), guess) != opponentWords.end()){
			color=1;
		}
		
		state->performGuess(guess, color, state->turn);
		bool opponentWon = true;
		for(string opponentWord : opponentWords){
			bool exists = false;
			for(string word : state->words){
				if(word == opponentWord){
					exists = true;
					break;
				}
			}
			if(exists){
				opponentWon = false;
				break;
			}
		}
		if(opponentWon){
#ifdef DEBUG_MCTS
			cerr << "Incorrect" << endl;
			cerr << "Opponent won" << endl;
#endif
			return 1;
		}
		vector<string>* goodWords = state->turn?&opponentWords:&myWords;
		if(find(all(*goodWords), guess) == goodWords->end()){
#ifdef DEBUG_MCTS
			cerr << "Incorrect" << endl;
#endif
			if(find(all(assassinWords), guess) != assassinWords.end()){
#ifdef DEBUG_MCTS
				cerr << "Assassin" << endl;
#endif
				return !state->turn;
			}
			return -2;
		}
#ifdef DEBUG_MCTS
		cerr << "Correct" << endl;
#endif
		bool IWon = true;
		for(string myWord : myWords){
			bool exists = false;
			for(string word : state->words){
				if(word == myWord){
					exists = true;
					break;
				}
			}
			if(exists){
				IWon = false;
				break;
			}
		}
		if(IWon){
#ifdef DEBUG_MCTS
			cerr << "I won" << endl;
#endif
			return 0;
		}
		return -1;
	}

	vector<string> intersection(vector<string> v1, vector<string> v2){
		vector<string> ret;
		rep(i,0,(int)v1.size()){
			rep(j,0,(int)v2.size()){
				if(v1[i] == v2[j]){
					ret.push_back(v1[i]);
					break;
				}
			}
		}
		return ret;
	}

	int simulate(State* state){
		Bot tmpBot(engine);
#ifdef DEBUG_MCTS
		cerr << "It is now player " << state->turn << "'s turn" << endl;
		state->printWords();
#endif
		if(state->turn == 0)
			tmpBot.setWords(
					intersection(myWords, state->words), 
					intersection(opponentWords, state->words), 
					intersection(greyWords, state->words), 
					intersection(assassinWords, state->words));
		else
			tmpBot.setWords(
					intersection(opponentWords, state->words), 
					intersection(myWords, state->words), 
					intersection(greyWords, state->words), 
					intersection(assassinWords, state->words));
		tmpBot.decentWords = decentWords;
		pair<string, int> clue = tmpBot.getBestWord(false, 0.2);
#ifdef DEBUG_MCTS
		cerr << "Selected the clue " << clue.first << " " << clue.second << endl;
#endif
		state->acceptClue(clue, state->turn, engine);
		rep(i,0,clue.second+1){
			int res = simulateGuessing(state);
			if(res == -2)
				break;
			if(res != -1)
				return res;
		}
		state->turn = !state->turn;
		return simulate(state);
	}

	struct ScoreEntry{
		int timesVisited;
		int score;

		ScoreEntry(){
			timesVisited = 0;
			score = 0;
		}

		float getScore(int totTimesVisited){
			if(!timesVisited)
				return 100;
			return (score+0.0)/timesVisited + 1.4*sqrt(log(totTimesVisited)/timesVisited);
		}
	};


	struct Node{
		vector<pair<pair<string, int>, ScoreEntry> > children;
		ScoreEntry score;

		Node(){
		}

	};

	map<State, Node*> stateMap;

	int rec(Node* node, State state, SimilarityEngine &engine){
#ifdef DEBUG_MCTS
		cerr << "It is now player " << state.turn << "'s turn" << endl;
		cerr << "In this state the win rate is " << node->score.score << "/" << node->score.timesVisited << endl;
		state.printWords();
#endif
		if(node->score.timesVisited > 0){
			node->score.timesVisited++;
			float bestScore;
			int bestInd;
			rep(i,0,(int)node->children.size()){
				float s = node->children[i].second.getScore(node->score.timesVisited);
				if(!i || s > bestScore){
					bestScore = s;
					bestInd = i;
				}
			}
			pair<string, int> clue = node->children[bestInd].first;
#ifdef DEBUG_MCTS
			cerr << "Selected the clue " << clue.first << " " << clue.second << endl;
			cerr << "This clue has win rate " << node->children[bestInd].second.score << "/" << node->children[bestInd].second.timesVisited << endl;
#endif
			State tmpState = state;
			tmpState.acceptClue(clue, state.turn, engine);
			rep(i,0,clue.second+1){
				int res = simulateGuessing(&tmpState);
				if(res == -2)
					break;
				if(res != -1){
					if(state.turn == res)
						++node->score.score;
					return res;
				}
			}
			tmpState.turn = !state.turn;
			auto it = stateMap.find(tmpState);
			if(it == stateMap.end()){
				stateMap[tmpState] = new Node();
				it = stateMap.find(tmpState);
			}
			int res = rec(it->second, tmpState, engine);
			if(state.turn == res){
				node->score.score++;
				node->children[bestInd].second.score++;
			}
			node->children[bestInd].second.timesVisited++;
			return res;
		}
		else{
			node->score.timesVisited++;
			Bot tmpBot(engine);
			if(state.turn == 0){
				tmpBot.setWords(
						intersection(myWords, state.words), 
						intersection(opponentWords, state.words), 
						intersection(greyWords, state.words), 
						intersection(assassinWords, state.words));
			}
			else{
				tmpBot.setWords(
						intersection(opponentWords, state.words), 
						intersection(myWords, state.words), 
						intersection(greyWords, state.words), 
						intersection(assassinWords, state.words));
			}
			tmpBot.decentWords = decentWords;
			vector<pair<string, int> > bestWords = tmpBot.getBestWords(5, false, 1.0);
			rep(i,0,(int)bestWords.size()){
				node->children.emplace_back(bestWords[i], ScoreEntry());
			}
			pair<string, int> clue = node->children[0].first;
#ifdef DEBUG_MCTS
			cerr << "Selected the clue " << clue.first << " " << clue.second << endl;
#endif
			State tmpState = state;
			tmpState.acceptClue(clue, state.turn, engine);
			rep(i,0,clue.second+1){
				int res = simulateGuessing(&tmpState);
				if(res == -2)
					break;
				if(res != -1){
					if(state.turn == res)
						++node->score.score;
					return res;
				}
			}
			tmpState.turn = !state.turn;
#ifdef DEBUG_MCTS
			cerr << "Leaving tree" << endl;
#endif
			int res = simulate(&tmpState);
			if(res == state.turn)
				node->score.score++;
			return res;
		}
	}

	pair<string, int> getBestWordMCTS(State curState) {
		getBestWord(false);
		vector<wordID> candidates = engine.getCommonWords(vocabularySize);
		decentWords.clear();
		rep(i,0,(int)candidates.size()){
			rep(j, 0, boardWords.size()) {
				char type = boardWords[j].type;
				float sim = engine.similarity(boardWords[j].id, candidates[i]);
				if((type == 'm' || type == 'o') && sim > 0.4){
					decentWords.push_back(candidates[i]);
					break;
				}
			}
		}
		cerr << "Number of decent words: " << decentWords.size() << "/" << candidates.size() << endl;
		Node root;
		rep(j,0,1000){
			rec(&root, curState, engine);
			if(j%500 == 0 && j > 0){
				cerr << "Stats after " << j << " simulations" << endl;
				rep(i,0,(int)root.children.size()){
					cerr << root.children[i].first.first << " " << root.children[i].first.second <<
						" " << root.children[i].second.score << "/" << root.children[i].second.timesVisited <<
						" = " << setprecision(4) << fixed << (0.0+root.children[i].second.score)/root.children[i].second.timesVisited << endl;
				}
			}
		}
		float bestScore;
		int bestInd;
		cerr << "Final stats" << endl;
		rep(i,0,(int)root.children.size()){
			float s = root.children[i].second.timesVisited;
			cerr << root.children[i].first.first << " " << root.children[i].first.second <<
				" " << root.children[i].second.score << "/" << root.children[i].second.timesVisited <<
						" = " << setprecision(4) << fixed << (0.0+root.children[i].second.score)/root.children[i].second.timesVisited << endl;
			if(!i || s > bestScore){
				bestScore = s;
				bestInd = i;
			}
		}
		pair<string, int> bestClue = root.children[bestInd].first;
		cerr << "Chose clue " << bestClue.first << " " << bestClue.second << endl;
		return bestClue;
	}
};

class GameInterface {
	SimilarityEngine &engine;
	Bot bot;
	vector<string> myWords, opponentWords, greyWords, assassinWords;
	string myColor;
	State curState;
	bool useMCTS = false;

	void commandReset() {
		myWords.clear();
		opponentWords.clear();
		greyWords.clear();
		assassinWords.clear();
		bot.setWords(myWords, opponentWords, greyWords, assassinWords);
		curState.reset();
	}

	void commandSuggestWord() {
		cout << "Thinking..." << endl;
		if(useMCTS)
			bot.getBestWordMCTS(curState);
		else
			bot.getBestWord(true);
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
			int color = -1;
			if(find(all(myWords), word) != myWords.end()){
				color=0;
			}
			else if(find(all(opponentWords), word) != opponentWords.end()){
				color=1;
			}
			eraseFromVector(word, myWords);
			eraseFromVector(word, opponentWords);
			eraseFromVector(word, greyWords);
			eraseFromVector(word, assassinWords);
			curState.performGuess(word, color, curState.turn);
		}

		if (v != NULL) {
			string word;
			cin >> word;
			word = toLowerCase(word);
			if (engine.wordExists(word)) {
				v->push_back(word);
				curState.addWord(word);
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

	void commandMCTS() {
		string state;
		cin >> state;
		state = toLowerCase(state);
		if(state == "on"){
			useMCTS = true;
			cerr << "Turning on MCTS" << endl;
		}
		else if(state == "off"){
			useMCTS = false;
			cerr << "Turning off MCTS" << endl;
		}
		else{
			cerr << "Cannot set useMCTS to " << state << ", use one of the value 'on' or 'off' instead" << endl;
		}
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
			} else if (command == "mcts") {
				commandMCTS();
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
	interface.run();
	return 0;
}
