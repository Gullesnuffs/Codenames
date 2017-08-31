#include <vector>
#include <string>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <set>
#include <map>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <queue>
#include <sstream>
using namespace std;

#define rep(i, a, b) for(int i = (a); i < int(b); ++i)
#define rrep(i, a, b) for(int i = (a) - 1; i >= int(b); --i)
#define trav(x, v) for(auto& x : v)
#define sz(x) (int)(x).size()
#define all(v) (v).begin(), (v).end()
#define what_is(x) cout << #x << " is " << x << endl;

typedef double fl;
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<pii> vpi;

// Sloppily and quickly parse a double
double parseDouble(const string& str) {
	double res = 0, factor = -1;
	int exp = 0;
	bool foundone = false, neg = false, parseexp = false;
	bool efoundone = false, eneg = false;
	trav(c, str) {
		if (parseexp) {
			if (c == '-') {
				assert(!efoundone && !eneg);
				eneg = true;
			} else {
				int dig = c - '0';
				assert(0 <= dig && dig < 10);
				efoundone = true;
				exp *= 10;
				exp += dig;
			}
		} else if (c == '-') {
			assert(!foundone && !neg && factor == -1);
			neg = true;
		} else if (c == '.') {
			assert(factor == -1);
			factor = 1;
		} else if (c == 'e' || c == 'E') {
			parseexp = true;
		} else {
			double dig = c - '0';
			assert(0 <= dig && dig < 10);
			if (factor == -1) {
				res *= 10;
				res += dig;
			} else {
				factor *= 0.1;
				res += factor * dig;
			}
			foundone = true;
		}
	}
	assert(foundone);
	if (neg) res = -res;
	if (parseexp) {
		assert(efoundone);
		if (eneg) exp = -exp;
		res *= pow(10, exp);
	}
	return res;
}

void processWord2Vec(const char* inFile, const char* popFile, const char* outFile, const char* wordlistFile, int modelid, int limit) {
	string line;
	set<string> wordlist;
	ifstream fin(wordlistFile);
	if (fin) {
		while (getline(fin, line)) {
			wordlist.insert(line);
		}
		fin.close();
	}
	else {
		cerr << "Warning: missing " << wordlistFile << ", so unable to ensure all words from there are available." << endl;
	}

	int popcount = 0; // (sorry)
	map<string, int> popularWords;
	trav(w, wordlist) popularWords[w] = -1;
	fin.open(popFile);
	assert(fin);
	while (getline(fin, line)) {
		istringstream iss(line);
		string word;
		iss >> word;
		assert(!popularWords.count(word) || wordlist.count(word));
		popularWords[word] = popcount++;
		if (sz(popularWords) == limit)
			break;
	}
	trav(w, wordlist) {
		if (popularWords[w] == -1)
			popularWords[w] = popcount++;
	}
	assert(sz(popularWords) == popcount);
	fin.close();

	struct Word {
		string word;
		float norm;
		vector<float> vec;
	};

	fin.open(inFile);
	assert(fin);
	vector<Word> words(popcount);
	int dim = -1, count = 0;
	while (getline(fin, line)) {
		size_t ind = line.find(' ');
		assert(ind != string::npos && ind != 0);
		Word w;
		w.word = line.substr(0, ind);
		auto indexit = popularWords.find(w.word);
		if (indexit == popularWords.end())
			continue;
		int index = indexit->second;
		popularWords.erase(indexit);

		double norm = 0;
		if (dim != -1)
			w.vec.reserve(dim);
		while (ind != string::npos) {
			size_t ind2 = line.find(' ', ind+1);
			string tok = line.substr(ind+1, ind2 == string::npos ? string::npos : ind2 - (ind+1));
			double x = parseDouble(tok);
			w.vec.push_back((float)x);
			norm += x*x;
			ind = ind2;
		}
		if (dim == -1) dim = sz(w.vec);
		else assert(sz(w.vec) == dim);

		w.norm = (float)norm;
		double mu = 1 / sqrt(norm);
		trav(x, w.vec) x = (float)(x * mu);
		wordlist.erase(w.word);
		words[index] = move(w);
		count++;
		if (count == popcount)
			break;
	}
	fin.close();

	if (!wordlist.empty()) {
		cerr << "Warning: words not found:" << endl;
		for (const string &w : wordlist)
			cerr << w << endl;
	}

	ofstream fout(outFile, ios::binary);
	int sentinel = -1;
	int version = 1;
	fout.write((char*)&sentinel, sizeof sentinel);
	fout.write((char*)&version, sizeof version);
	fout.write((char*)&modelid, sizeof modelid);
	fout.write((char*)&count, sizeof count);
	fout.write((char*)&dim, sizeof dim);
	trav(w, words) {
		int len = sz(w.word);
		if (!len) continue;
		fout.write((char*)&len, sizeof len);
		fout.write(w.word.data(), len);
		fout.write((char*)&w.norm, sizeof(float));
		fout.write((char*)w.vec.data(), dim * sizeof(float));
	}
	fout.close();
}

int main(int argc, char **argv) {
	if (argc != 6) {
		cerr << "Usage: " << argv[0] << " <word2vec .txt file> <popularity .txt file> <model id> <limit> <outfile.bin>" << endl;
		cerr << endl;
		cerr << "* The word2vec file should be a list of lines of the form \"word a_1 a_2 ... a_k\"," << endl;
		cerr << " where k is the dimension of the word2vec embedding, a_i are real numbers in decimal form," << endl;
		cerr << " and words are lower-case with spaces replaced by underscores." << endl;
		cerr << endl;
		cerr << "* The popularity file contains words to be included, in order of decreasing commonness." << endl;
		cerr << " Only the first token of every line is considered; thus, word2vec txt files can be used here as well." << endl;
		cerr << endl;
		cerr << "* Additionally, if wordlist.txt exists, it is prepended to the popularity file." << endl;
		cerr << " It is intended to contain all the words from the game." << endl;
		cerr << endl;
		cerr << "* The model id is an arbitrary integer representing the model." << endl;
		cerr << endl;
		cerr << "* The limit indicates the number of words from the popularity file to use. 0 = unlimited." << endl;
		cerr << " Around 50,000 is reasonable." << endl;
		return 1;
	}

	const char* inFile = argv[1];
	const char* popFile = argv[2];
	int modelid = atoi(argv[3]);
	int limit = atoi(argv[4]);
	const char* outFile = argv[5];
	processWord2Vec(inFile, popFile, outFile, "wordlist.txt", modelid, limit);
}
