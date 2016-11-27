#include <bits/stdc++.h>
using namespace std;

#define rep(i, a, b) for(int i = (a); i < int(b); ++i)
#define rrep(i, a, b) for(int i = (a) - 1; i >= int(b); --i)
#define trav(it, v) for(typeof((v).begin()) it = (v).begin(); it != (v).end(); ++it)
#define all(v) (v).begin(), (v).end()
#define what_is(x) cout << #x << " is " << x << endl;

typedef double fl;
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
			s[i] += 'a'-'A';
	return s;
}

double sigmoid(double x){
	return 1.0/(1.0+exp(-x));
}

struct SimilarityEngine{

	map<string, vector<double> > vec;
	map<string, int> popularity;

	// Returns if successful
	bool load(const char* fileName, int numberOfWords){
		FILE* fin = fopen(fileName, "r");
		if(fin == NULL)
			return false;
		char buffer[1<<16];
		while(fscanf(fin, "%s", buffer) != EOF){
			vector<double> v;
			double length=0;
			rep(i,0,300){
				double x;
				ignore = fscanf(fin, "%lf", &x);
				v.push_back(x);
				length += x*x;
			}
			length=sqrt(length);
			rep(i,0,v.size()){
				v[i] /= length;
			}
			vec[string(buffer)]=v;
			popularity[string(buffer)]=vec.size();

			// If numberOfWords is non-positive it is assumed that we should load all words
			if(numberOfWords > 0){
				if(vec.size() == (((10*vec.size())/numberOfWords)*numberOfWords)/10)
					cout << ((100LL*((int)vec.size()))/numberOfWords) << " %" << endl;
				if(vec.size() == numberOfWords)
					break;
			}
		}
		return true;
	}

	vector<string> getCommonWords(int vocabularySize){
		vector<string> ret;
		for(auto entry : popularity)
			if(entry.second < vocabularySize)
				ret.push_back(entry.first);
		return ret;
	}

	double similarity(string s1, string s2){
		if(!vec.count(s1)){
			cout << s1 << " does not occur in the corpus" << endl;
			return 0;
		}
		if(!vec.count(s2)){
			cout << s2 << " does not occur in the corpus" << endl;
			return 0;
		}
		vector<double>* v1=&vec[s1];
		vector<double>* v2=&vec[s2];
		double ret=0;
		rep(i,0,v1->size()){
			ret += (*v1)[i]*(*v2)[i];
		}
		return ret;
	}

	vector<pair<double, string> > similarWords(string s){
		if(!vec.count(s)){
			cout << s << " does not occur in the corpus" << endl;
			return vector<pair<double, string> >();
		}
		vector<pair<double, string> > ret;
		for(auto it=vec.begin(); it != vec.end(); ++it){
			ret.push_back(make_pair(-similarity(s, it->first), it->first));
		}
		sort(all(ret));
		vector<pair<double, string> > res;
		rep(i,0,10)
			res.push_back(make_pair(-ret[i].first, ret[i].second));
		return res;
	}

};

struct Bot{

	// Give a similarity bonus to "bad" words
	double marginOpponentWords = 0.02;
	double marginAssassins = 0.05;

	// Constants used in scoring function based
	// on the sigmoid function of the similarities
	double fuzzyWeightAssassin = -1.0;
	double fuzzyWeightOpponent = -0.5;
	double fuzzyWeightMy = 0.5;
	double fuzzyWeightGrey = -0.1;
	double fuzzyExponent = 15;
	double fuzzyOffset = 0.3;

	// Assume that we will never succeed if the similarity
	// is at most minSimilarity
	double minSimilarity = 0.2;

	// How bad is it if there is an opponent word with high similarity
	double weightOpponent = -1.5;
	// How bad is it if there is a grey word with high similarity
	double weightGrey = -0.2;
	
	// How important is it that the last good word has greater
	// similarity than the next bad word
	double marginWeight = 0.2;

	// Number of words that are considered common
	int commonWordLimit = 5000;

	// Prefer common words
	double commonWordWeight = 1.2;

	// Number of words that are not considered rare
	int rareWordLimit = 15000;

	// Avoid rare words
	double rareWordWeight = 0.8;

	// Consider only the 50000 most common words
	int vocabularySize = 50000;

	vector<string> myWords, opponentWords, greyWords, assassinWords;
	vector<pair<char, string> > boardWords;

	pair<double, int> getWordScore(string word, bool debugPrint, SimilarityEngine& engine){
		if(debugPrint)
			cout << "Printing statistics for \"" << word << "\"" << endl;

		// Check if word is a substring or a superstring of any of the words on the board
		rep(i,0,boardWords.size()){
			if(boardWords[i].second.find(word) != string::npos)
				return make_pair(-1000, -1);
			if(word.find(boardWords[i].second) != string::npos)
				return make_pair(-1000, -1);
		}

		vector<pair<double, pair<char, string> > > v;
		rep(i,0,boardWords.size()){
			double sim=engine.similarity(boardWords[i].second, word);
			if(boardWords[i].first == 'o')
				sim += marginOpponentWords;
			if(boardWords[i].first == 'a')
				sim += marginAssassins;
			v.push_back(make_pair(-sim, boardWords[i]));
		}
		sort(all(v));
		double bestScore=0;
		int bestCount=0;
		double curScore=0;
		double lastGood=0;
		double baseScore=0;
		int curCount=0;
		if(debugPrint){
			rep(i,0,v.size()){
				cout << -v[i].first << "\t" << v[i].second.second << " ";
				switch(v[i].second.first){
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
			double weight;
			if(v[i].second.first == 'a')
				weight = fuzzyWeightAssassin;
			if(v[i].second.first == 'o')
				weight = fuzzyWeightOpponent;
			if(v[i].second.first == 'm')
				weight = fuzzyWeightMy;
			if(v[i].second.first == 'g')
				weight = fuzzyWeightGrey;
			double contribution = weight * sigmoid((-v[i].first - fuzzyOffset) * fuzzyExponent);
			baseScore += contribution;
		}

		rep(i,0,v.size()){
			if(-v[i].first < minSimilarity)
				break;
			if(v[i].second.first == 'a')
				break;
			if(v[i].second.first == 'o'){
				curScore += weightOpponent;
				continue;
			}
			if(v[i].second.first == 'm'){
				lastGood=-v[i].first;
				curScore += sigmoid((-v[i].first - fuzzyOffset) * fuzzyExponent);
				++curCount;
			}
			if(v[i].second.first == 'g'){
				curScore += weightGrey;
				continue;
			}
			double tmpScore=-1;
			rep(j,i+1,v.size()){
				if(v[j].second.first == 'a' || v[j].second.first == 'o'){
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
		if(engine.popularity[word] < commonWordLimit)
			bestScore *= commonWordWeight;
		else if(engine.popularity[word] > rareWordLimit)
			bestScore *= rareWordWeight;
		return make_pair(bestScore, bestCount);
	}

	void createBoardWords(){
		boardWords.clear();
		rep(i,0,myWords.size())
			boardWords.push_back(make_pair('m', myWords[i]));
		rep(i,0,opponentWords.size())
			boardWords.push_back(make_pair('o', opponentWords[i]));
		rep(i,0,greyWords.size())
			boardWords.push_back(make_pair('g', greyWords[i]));
		rep(i,0,assassinWords.size())
			boardWords.push_back(make_pair('a', assassinWords[i]));
	}
	
	pair<string, int> getBestWord(SimilarityEngine& engine, vector<string> _myWords, vector<string> _opponentWords, 
			vector<string> _greyWords, vector<string> _assassinWords){
		myWords = _myWords;
		opponentWords = _opponentWords;
		greyWords = _greyWords;
		assassinWords = _assassinWords;
		createBoardWords();
		vector<pair<double, pair<int, string> > > v;
		string bestWord="";
		int bestCount;
		double bestScore=0;
		vector<string> candidates = engine.getCommonWords(vocabularySize);
		for(string candidate : candidates){
			pair<double, int> res=getWordScore(candidate, false, engine);
			v.push_back(make_pair(-res.first, make_pair(res.second, candidate)));
			if(res.first > bestScore){
				bestScore=res.first;
				bestCount=res.second;
				bestWord=candidate;
			}
		}
		sort(all(v));
		// Print how the score of the best word was computed
		getWordScore(bestWord, true, engine);

		// Print a list with the best clues
		rep(i,0,20){
			pair<double, int> res=getWordScore(v[i].second.second, false, engine);
			cout << (i+1) << "\t" << setprecision(3) << fixed << res.first << "\t" << v[i].second.second << " " << res.second << endl;
		}
		int p=engine.popularity[bestWord];
		cout << "The best clue found is " << bestWord << " " << bestCount << endl;
		cout << bestWord << " is the " << p;
		if(p % 10 == 1 && p % 100 != 11)
			cout << "st";
		else if(p % 10 == 2 && p % 100 != 12)
			cout << "nd";
		else if(p % 10 == 3 && p % 100 != 13)
			cout << "rd";
		else
			cout << "th";
		cout << " most popular word" << endl;
		return make_pair(bestWord, bestCount);
	}
};

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

int main(){
	SimilarityEngine engine;
	engine.load("data.txt", 100000);
	cout << "Finished loading!" << endl;
	cout << "Type \"help\" for help" << endl;
	cout << "My color (b/r): ";
	string myColor = inputColor();
	Bot bot;
	vector<string> myWords, opponentWords, greyWords, assassinWords;
	while(true){
		string command1;
		cin >> command1;
		command1 = toLowerCase(command1);
		vector<string>* v = NULL;
		if(command1 == myColor)
			v=&myWords;
		else if(command1 == "b" || command1 == "r")
			v=&opponentWords;
		else if(command1 == "g" || command1 == "c")
			v=&greyWords;
		else if(command1 == "a")
			v=&assassinWords;
		else if(command1 == "-"){
			string word;
			cin >> word;
			word=toLowerCase(word);
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
			}
			else{
				cout << word << " was not found in the dictionary" << endl;
			}
		}
		if(command1 == "play" || command1 == "go"){
			cout << "Thinking..." << endl;
			pair<string, int> best = bot.getBestWord(engine, myWords, opponentWords, greyWords, assassinWords);
		}
		if(command1 == "reset"){
			myWords.clear();
			opponentWords.clear();
			greyWords.clear();
			assassinWords.clear();
		}
		if(command1 == "help" || command1 == "\"help\""){
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
		if(command1 == "board"){
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
	}
}
