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

void processWord2Vec(const char* inFile, const char* outFile, const char* wordlistFile, int limit) {
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
		cerr << "Warning: missing wordlist.txt, so unable to ensure all words from there are available." << endl;
	}

	struct Word {
		string word;
		vector<float> vec;
	};

	fin.open(inFile);
	assert(fin);
	vector<Word> words;
	int dim = -1, count = 0;
	while (getline(fin, line)) {
		istringstream iss(line);
		Word w;
		iss >> w.word;
		if (count >= limit && !wordlist.count(w.word))
			continue;
		double length = 0;
		double x;
		if (dim != -1)
			w.vec.reserve(dim);
		while (iss >> x) {
			w.vec.push_back((float)x);
			length += x*x;
		}
		if (dim == -1) dim = sz(w.vec);
		else assert(sz(w.vec) == dim);
		length = 1 / sqrt(length);
		trav(x, w.vec) x = (float)(x * length);
		words.push_back(w);
		wordlist.erase(w.word);
		count++;
		if (count >= limit && wordlist.empty())
			break;
	}
	fin.close();

	if (!wordlist.empty()) {
		cerr << "Warning: words not found:" << endl;
		for (const string &w : wordlist)
			cerr << w << endl;
	}

	ofstream fout(outFile, ios::binary);
	fout.write((char*)&count, sizeof count);
	fout.write((char*)&dim, sizeof dim);
	trav(w, words) {
		int len = sz(w.word);
		fout.write((char*)&len, sizeof len);
		fout.write(w.word.data(), w.word.size());
		fout.write((char*)w.vec.data(), dim * sizeof(float));
	}
	fout.close();
}

int main(int argc, char **argv) {
	if (argc != 3) {
		cerr << "Usage: " << argv[0] << " <word2vec .txt file> <limit>" << endl;
		return 1;
	}
	const char* inFile = argv[1];
	int limit = atoi(argv[2]);
	processWord2Vec(inFile, "data.bin", "wordlist.txt", limit);
}
