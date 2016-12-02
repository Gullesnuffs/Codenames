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

void processWord2Vec(const char* inFile, const char* outFile, int weightLowerBound) {
	map<string, int> word2id;
	vector<string> words;
	vector<pair<pii, float>> edges;

	int counter = 0;
	auto fin = ifstream(inFile, ios::binary);
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
			cerr << "Read " << counter << " entries with " << words.size() << " words" << endl;
			break;
		}

		if (weight < weightLowerBound) continue;

		counter++;
		if ((counter % 100000) == 0) {
			cerr << "\rRead " << counter << " entries...";
		}

		auto it1 = word2id.find(word1);
		if (it1 == word2id.end()) {
			it1 = word2id.insert({word1, words.size()}).first;
			words.push_back(word1);
		}

		auto it2 = word2id.find(word2);
		if (it2 == word2id.end()) {
			it2 = word2id.insert({word2, words.size()}).first;
			words.push_back(word2);
		}
		
		auto id1 = it1->second;
		auto id2 = it2->second;

		edges.push_back({{id1,id2}, 1.0 / sqrt(weight) + 0.1});
		edges.push_back({{id2,id1}, 10.0 / sqrt(weight) + 0.5});
	}

	ofstream fout(outFile, ios::binary);
	int numWords = words.size();
	fout.write((char*)&numWords, sizeof numWords);
	for (auto word : words) {
		int len = sz(word);
		fout.write((char*)&len, sizeof(len));
		fout.write(word.data(), word.size());
	}
	int numEdges = edges.size();
	fout.write((char*)&numEdges, sizeof(numEdges));
	for (auto edge : edges) {
		fout.write((char*)&edge.first.first, sizeof(int));
		fout.write((char*)&edge.first.second, sizeof(int));
		fout.write((char*)&edge.second, sizeof(float));
	}
	fout.close();
}

int main(int argc, char **argv) {
	if (argc != 3) {
		cerr << "Usage: " << argv[0] << " <concepts .txt file> <weightLowerBound>" << endl;
		return 1;
	}
	const char* inFile = argv[1];
	int weightLowerBound = atoi(argv[2]);
	processWord2Vec(inFile, "concept_data.bin", weightLowerBound);
}
