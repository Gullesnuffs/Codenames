#include <bits/stdc++.h>
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

void processWord2Vec(const char* inFile, const char* outFile, int limit) {
	struct Word {
		string word;
		vector<float> vec;
	};

	ifstream fin(inFile);
	fin.exceptions(fin.badbit | fin.failbit);
	assert(fin);
	vector<Word> words;
	string line;
	int dim = -1, count = 0;
	while (getline(fin, line)) {
		istringstream iss(line);
		Word w;
		iss >> w.word;
		double length = 0;
		double x;
		if (dim != -1)
			w.vec.reserve(dim);
		while (iss >> x) {
			w.vec.push_back(x);
			length += x*x;
		}
		if (dim == -1) dim = sz(w.vec);
		else assert(sz(w.vec) == dim);
		length = 1 / sqrt(length);
		trav(x, w.vec) x *= length;
		words.push_back(w);
		count++;
		if (count == limit) break;
	}
	fin.close();

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

int main() {
	processWord2Vec("data.txt", "data.bin", 100000);
}
