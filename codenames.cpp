#include <bits/stdc++.h>
using namespace std;

#define rep(i, a, b) for(int i = (a); i < int(b); ++i)
#define rrep(i, a, b) for(int i = (a) - 1; i >= int(b); --i)
#define trav(x, v) for(auto& x : v)
#define all(v) (v).begin(), (v).end()
#define what_is(x) cout << #x << " is " << x << endl;

typedef float fl;
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<pii> vpi;

void eraseFromVector(string word, vector<string>* v){
	rep(i,0,v->size())
		if((*v)[i] == word)
			v->erase(v->begin()+i);
}

string toLowerCase(string s){
	rep(i,0,s.size())
		if(s[i] >= 'A' && s[i] <= 'Z')
			s[i] = (char)(s[i] + 'a' - 'A');
	return s;
}

fl sigmoid(fl x){
	return 1.0f/(1.0f+exp(-x));
}

struct SimilarityEngine{

	map<string, vector<fl> > vec;
	map<string, int> popularity;

	// Returns if successful
	bool load(const char* fileName) {
		int dimension, numberOfWords;
		ifstream fin(fileName, ios::binary);
		fin.read((char*)&numberOfWords, sizeof numberOfWords);
		fin.read((char*)&dimension, sizeof dimension);
		if (!fin) {
			cerr << "Failed to load " << fileName << endl;
			return false;
		}
		cerr << "Loading word2vec (" << numberOfWords << " words, " << dimension << " dimensions)..." << flush;

		const int bufSize = 1 << 16;
		char buf[bufSize];
		string word;
		vector<float> values(dimension);
		vector<fl> valuesd;
		rep(i,0,numberOfWords) {
			int len;
			fin.read((char*)&len, sizeof len);
			if (!fin) {
				cerr << " failed at reading entry " << i << endl;
				return false;
			}
			if (len > bufSize || len <= 0) {
				cerr << " invalid length " << len << endl;
				return false;
			}
			fin.read(buf, len);
			fin.read((char*)values.data(), dimension * sizeof(float));
			if (!fin) {
				cerr << " failed at reading entry " << i << endl;
				return false;
			}
			word.assign(buf, buf + len);
			valuesd.assign(all(values));
			vec[word] = move(valuesd);
			popularity[word] = i + 1;
		}
		cerr << " done!" << endl;
		return true;
	}

	vector<const string*> getCommonWords(int vocabularySize){
		vector<const string*> ret;
		ret.reserve(vocabularySize);
		for(const auto& entry : popularity)
			if(entry.second < vocabularySize)
				ret.push_back(&entry.first);
		return ret;
	}

	fl similarity(const vector<fl>& v1, const vector<fl>& v2) {
		fl ret=0;
		int dim = (int)v1.size();
		rep(i,0,dim) {
			ret += v1[i] * v2[i];
		}
		return ret;
	}

	fl similarity(const string& s1, const string& s2){
		return similarity(vec.at(s1), vec.at(s2));
	}

	const vector<fl>& getVec(const string& s) {
		return vec.at(s);
	}

	vector<pair<fl, string> > similarWords(string s){
		if(!vec.count(s)){
			cout << s << " does not occur in the corpus" << endl;
			return vector<pair<fl, string> >();
		}
		vector<pair<fl, string> > ret;
		for(auto it=vec.begin(); it != vec.end(); ++it){
			ret.push_back(make_pair(-similarity(s, it->first), it->first));
		}
		sort(all(ret));
		vector<pair<fl, string> > res;
		rep(i,0,10)
			res.push_back(make_pair(-ret[i].first, ret[i].second));
		return res;
	}

};

struct Bot{

	// Give a similarity bonus to "bad" words
	fl marginOpponentWords = 0.02f;
	fl marginAssassins = 0.05f;

	// Constants used in scoring function based
	// on the sigmoid function of the similarities
	fl fuzzyWeightAssassin = -1.0f;
	fl fuzzyWeightOpponent = -0.5f;
	fl fuzzyWeightMy = 0.5f;
	fl fuzzyWeightGrey = -0.1f;
	fl fuzzyExponent = 15;
	fl fuzzyOffset = 0.3f;

	// Assume that we will never succeed if the similarity
	// is at most minSimilarity
	fl minSimilarity = 0.2f;

	// How bad is it if there is an opponent word with high similarity
	fl weightOpponent = -1.5f;
	// How bad is it if there is a grey word with high similarity
	fl weightGrey = -0.2f;

	// How important is it that the last good word has greater
	// similarity than the next bad word
	fl marginWeight = 0.2f;

	// Number of words that are considered common
	int commonWordLimit = 1000;

	// Prefer common words
	fl commonWordWeight = 0.9f;

	// Number of words that are not considered rare
	int rareWordLimit = 15000;

	// Avoid rare words
	fl rareWordWeight = 0.8f;

	// Consider only the 50000 most common words
	int vocabularySize = 50000;

	SimilarityEngine& engine;

	Bot(SimilarityEngine& engine) : engine(engine) {}

	vector<string> myWords, opponentWords, greyWords, assassinWords;
	struct BoardWord {
		char type;
		string word;
		vector<fl> vec;
	};
	vector<BoardWord> boardWords;
	void addBoardWord(char type, const string& word) {
		boardWords.push_back({type, word, engine.getVec(word)});
	}

	pair<fl, int> getWordScore(const string& word, bool debugPrint) {
		if(debugPrint)
			cout << "Printing statistics for \"" << word << "\"" << endl;

		// Check if word is a substring or a superstring of any of the words on the board
		rep(i,0,boardWords.size()){
			if(toLowerCase(boardWords[i].word).find(toLowerCase(word)) != string::npos)
				return make_pair(-1000, -1);
			if(toLowerCase(word).find(toLowerCase(boardWords[i].word)) != string::npos)
				return make_pair(-1000, -1);
		}

		const vector<fl>& wordVec = engine.getVec(word);

		typedef pair<fl, BoardWord*> Pa;
		static vector<Pa> v;
		v.clear();
		rep(i,0,boardWords.size()){
			fl sim=engine.similarity(boardWords[i].vec, wordVec);
			if(boardWords[i].type == 'o')
				sim += marginOpponentWords;
			if(boardWords[i].type == 'a')
				sim += marginAssassins;
			v.push_back(make_pair(-sim, &boardWords[i]));
		}
		sort(all(v), [&](const Pa& a, const Pa& b) { return a.first < b.first; });
		fl bestScore=0;
		int bestCount=0;
		fl curScore=0;
		fl lastGood=0;
		fl baseScore=0;
		int curCount=0;
		if(debugPrint){
			rep(i,0,v.size()){
				cout << setprecision(6) << fixed << -v[i].first << "\t" << v[i].second->word << " ";
				switch(v[i].second->type){
					case 'm': cout << "(My)" << endl; break;
					case 'o': cout << "(Opponent)" << endl; break;
					case 'g': cout << "(Civilian)" << endl; break;
					case 'a': cout << "(Assassin)" << endl; break;
					default: assert(0);
				}
			}
			cout << endl;
		}

		// Compute a fuzzy score
		rep(i,0,v.size()){
			char type = v[i].second->type;
			fl weight;
			if (type == 'a') weight = fuzzyWeightAssassin;
			else if (type == 'o') weight = fuzzyWeightOpponent;
			else if (type == 'm') weight = fuzzyWeightMy;
			else if (type == 'g') weight = fuzzyWeightGrey;
			else assert(0);
			fl contribution = weight * sigmoid((-v[i].first - fuzzyOffset) * fuzzyExponent);
			baseScore += contribution;
		}

		rep(i,0,v.size()){
			if(-v[i].first < minSimilarity)
				break;
			char type = v[i].second->type;
			if(type == 'a')
				break;
			if(type == 'o'){
				curScore += weightOpponent;
				continue;
			}
			if(type == 'm'){
				lastGood=-v[i].first;
				curScore += sigmoid((-v[i].first - fuzzyOffset) * fuzzyExponent);
				++curCount;
			}
			if(type == 'g'){
				curScore += weightGrey;
				continue;
			}
			fl tmpScore=-1;
			rep(j,i+1,v.size()){
				char type2 = v[j].second->type;
				if(type2 == 'a' || type2 == 'o') {
					tmpScore = marginWeight * sigmoid((lastGood - (-v[j].first)) * fuzzyExponent);
					break;
				}
			}
			tmpScore += baseScore + curScore;
			if(tmpScore > bestScore){
				bestScore=tmpScore;
				bestCount=curCount;
			}
		}

		int popularity = engine.popularity.at(word);
		if(popularity < commonWordLimit)
			bestScore *= commonWordWeight;
		else if(popularity > rareWordLimit)
			bestScore *= rareWordWeight;
		return make_pair(bestScore, bestCount);
	}

	void createBoardWords(){
		boardWords.clear();
		trav(w, myWords) addBoardWord('m', w);
		trav(w, opponentWords) addBoardWord('o', w);
		trav(w, greyWords) addBoardWord('g', w);
		trav(w, assassinWords) addBoardWord('a', w);
	}

	/** Returns suffix on number such as 'th' for 5 or 'nd' for 2 */
	string orderSuffix(int p) {
		if(p % 10 == 1 && p % 100 != 11) {
			return "st";
		} else if(p % 10 == 2 && p % 100 != 12) {
			return "nd";
		} else if(p % 10 == 3 && p % 100 != 13) {
			return "rd";
		} else {
			return "th";
		}
	}

	pair<string, int> getBestWord(
			const vector<string>& _myWords, const vector<string>& _opponentWords,
			const vector<string>& _greyWords, const vector<string>& _assassinWords){
		myWords = _myWords;
		opponentWords = _opponentWords;
		greyWords = _greyWords;
		assassinWords = _assassinWords;
		createBoardWords();
		vector<pair<fl, pair<int, const string*>>> wordScores;
		const string* bestWord=0;
		int bestCount=-1;
		fl bestScore=0;
		vector<const string*> candidates = engine.getCommonWords(vocabularySize);
		wordScores.reserve(candidates.size());
		for (const string* candidate : candidates){
			pair<fl, int> res = getWordScore(*candidate, false);
			wordScores.push_back(make_pair(-res.first, make_pair(res.second, candidate)));

			if (res.first > bestScore){
				bestScore=res.first;
				bestCount=res.second;
				bestWord=candidate;
			}
		}

		assert(bestWord);
		sort(all(wordScores));
		// Print how the score of the best word was computed
		getWordScore(*bestWord, true);

		// Print a list with the best clues
		rep(i,0,20){
			pair<fl, int> res = getWordScore(*wordScores[i].second.second, false);
			cout << (i+1) << "\t" << setprecision(3) << fixed << res.first << "\t" << *wordScores[i].second.second << " " << res.second << endl;
		}

		int p = engine.popularity.at(*bestWord);
		cout << "The best clue found is " << *bestWord << " " << bestCount << endl;
		cout << *bestWord << " is the " << p << orderSuffix(p);
		
		cout << " most popular word" << endl;
		return make_pair(*bestWord, bestCount);
	}
};

class GameInterface {
	SimilarityEngine engine;
	Bot bot;
	vector<string> myWords, opponentWords, greyWords, assassinWords;
	string myColor;

	void commandReset() {
		myWords.clear();
		opponentWords.clear();
		greyWords.clear();
		assassinWords.clear();
	}

	void commandSuggestWord() {
		cout << "Thinking..." << endl;
		pair<string, int> best = bot.getBestWord(myWords, opponentWords, greyWords, assassinWords);
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
	}

	void commandBoard() {
		cout << "My spies:";
		for(auto word : myWords){
			cout << " " << word;
		}
		cout << endl;
		cout << "Opponent spies:";
		for(auto word : opponentWords){
			cout << " " << word;
		}
		cout << endl;
		cout << "Civilians:";
		for(auto word : greyWords){
			cout << " " << word;
		}
		cout << endl;
		cout << "Assassins:";
		for(auto word : assassinWords){
			cout << " " << word;
		}
		cout << endl;
	}

	void commandModifyBoard(string command) {
		vector<string>* v = NULL;
		if(command == myColor) {
			v = &myWords;
		} else if(command == "b" || command == "r") {
			v = &opponentWords;
		} else if(command == "g" || command == "c") {
			v = &greyWords;
		} else if(command == "a") {
			v = &assassinWords;
		} else if(command == "-"){
			string word;
			cin >> word;
			word = toLowerCase(word);
			eraseFromVector(word, &myWords);
			eraseFromVector(word, &opponentWords);
			eraseFromVector(word, &greyWords);
			eraseFromVector(word, &assassinWords);
		}

		if(v != NULL){
			string word;
			cin >> word;
			word=toLowerCase(word);
			if(engine.popularity.count(word)){
				v->push_back(word);
			} else {
				cout << word << " was not found in the dictionary" << endl;
			}
		}
	}

	string inputColor(){
		string color;
		cin >> color;
		color = toLowerCase(color);
		while(true){
			if(color == "b" || color == "blue"){
				color = "b";
				break;
			}
			if(color == "r" || color == "red"){
				color = "r";
				break;
			}
			cin >> color;
			color = toLowerCase(color);
		}
		return color;
	}

public:
	GameInterface () : bot(engine) {
	}
	
	void run() {
		if (!engine.load("data.bin")) {
			cerr << "Failed to load data.bin" << endl;
			return;
		}

		cout << "Type \"help\" for help" << endl;
		cout << "My color (b/r): ";
		myColor = inputColor();
		
		while(true){
			string command1;
			cin >> command1;
			if (!cin) break;
			command1 = toLowerCase(command1);

			if (command1.size() == 1 && string("rgbac").find(command1) != string::npos) {
				commandModifyBoard(command1);
			}

			if(command1 == "play" || command1 == "go"){
				commandSuggestWord();
			}

			if(command1 == "quit" || command1 == "exit") {
				break;
			}

			if(command1 == "reset"){
				commandReset();
			}

			if(command1 == "help" || command1 == "\"help\""){
				commandHelp();
			}

			if(command1 == "board"){
				commandBoard();
			}
		}
	}
};

int main(){
	GameInterface interface;
	interface.run();
	return 0;
}
