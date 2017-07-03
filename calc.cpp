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

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define rrep(i, a, b) for (int i = (a)-1; i >= int(b); --i)
#define trav(x, v) for (auto &x : v)
#define all(v) (v).begin(), (v).end()
#define what_is(x) cout << #x << " is " << x << endl;

typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<pii> vpi;

enum wordID : int {};

struct SimilarityEngine {
	int dimension, formatVersion, modelid;
	map<string, wordID> word2id;
	vector<vector<float>> words;
	vector<string> wordsStrings;
	enum Models {
		GLOVE = 1, CONCEPTNET = 2
	};

	// All word vectors are stored normalized -- wordNorms holds their original norms.
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
	bool load(const char *fileName, bool quiet) {
		int numberOfWords;
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
		if (!quiet) {
			cerr << "Loading word2vec (" << numberOfWords << " words, "
				<< dimension << " dimensions, model " << modelid
				<< '.' << formatVersion << ")... " << flush;
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
		if (!quiet) {
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
		double sim = similarity(words[fixedWord], words[dynWord]);
		if (modelid == Models::GLOVE) {
			return sim * wordNorms[dynWord] / 4.5;
		} else if (modelid == Models::CONCEPTNET) {
			return (sim <= 0 ? sim : pow(sim, 0.8) * 1.6);
		} else {
			return sim;
		}
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

	vector<pair<float, string>> similarWords(const vector<float> &vec) {
		vector<pair<float, wordID>> ret;
		rep(i, 0, (int)words.size()) {
			ret.push_back(make_pair(-similarity(vec, words[i]), wordID(i)));
		}
		sort(all(ret));
		vector<pair<float, string>> res;
		rep(i, 0, 10) {
			res.push_back(make_pair(-ret[i].first, getWord(ret[i].second)));
		}
		return res;
	}
};

string trim(string w) {
	size_t ind = w.find_first_not_of(" ");
	w = (ind == string::npos ? "" : w.substr(ind));
	reverse(all(w));
	ind = w.find_first_not_of(" ");
	w = (ind == string::npos ? "" : w.substr(ind));
	reverse(all(w));
	return w;
}

double parse(string w) {
	istringstream iss(w);
	double res = 1;
	iss >> res;
	return res;
}

bool eval(SimilarityEngine &engine, const string &line, vector<float> *out) {
	size_t i = 0;
	vector<pair<double, string>> stuff;
	char lastSign = '+';
	for (;;) {
		size_t j = line.find_first_of("+-", i);
		string sub = trim(j == string::npos ? line.substr(i) : line.substr(i, j - i));
		size_t k = sub.find_first_not_of(".0123456789");
		if (k == string::npos) {
			cout << "missing word" << endl;
			return false;
		}
		string dec = sub.substr(0, k);
		string w = trim(sub.substr(k));
		if (!engine.wordExists(w)) {
			cout << "unknown word " << w << endl;
			return false;
		}
		double val = parse(dec);
		if (lastSign == '-') val = -val;
		stuff.push_back({val, w});
		if (j == string::npos) break;
		lastSign = line[j];
		i = j+1;
	}

	int dim = engine.dimension;
	vector<float> vec(dim);
	trav(pa, stuff) {
		vector<float> vec2 = engine.words[engine.getID(pa.second)];
		rep(i,0,dim) {
			vec[i] += pa.first * vec2[i];
		}
	}
	*out = move(vec);
	return true;
}

struct Color {
    double r,g,b;
};

// From https://stackoverflow.com/questions/7706339/grayscale-to-red-green-blue-matlab-jet-color-scale
Color getColor(double v,double vmin,double vmax) {
   Color c = {1.0,1.0,1.0};

   if (v < vmin)
      v = vmin;
   if (v > vmax)
      v = vmax;
   double dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      c.r = 0;
      c.g = 4 * (v - vmin) / dv;
   } else if (v < (vmin + 0.5 * dv)) {
      c.r = 0;
      c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
   } else if (v < (vmin + 0.75 * dv)) {
      c.r = 4 * (v - vmin - 0.5 * dv) / dv;
      c.b = 0;
   } else {
      c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      c.b = 0;
   }

   return c;
}

int ci(double x) { return (int)round(x*255); }
void printWithColor(string c, Color col) {
	cout << "\x1b[38;2;" << ci(col.r) << ';' << ci(col.g) << ';' << ci(col.b) << 'm' << c << "\x1b[0m";
}

int main() {
	SimilarityEngine engine;
	engine.load("data.bin", false);
	const int dim = engine.dimension;
	for (;;) {
		string line;
		cout << "> ";
		getline(cin, line);
		if (!cin) break;
		if (line.empty()) continue;
		vector<float> vec;
		size_t i = line.find('*');
		if (i == string::npos) {
			if (!eval(engine, line, &vec)) continue;

			float max = 0, min = 0;
			rep(i,0,dim) {
				max = std::max(max, vec[i]);
				min = std::min(min, vec[i]);
			}
			max = std::max(max, -min);
			if (max == 0) max = 1;
			rep(i,0,dim) {
				float v = vec[i] / max;
				Color c = (v < 0 ? Color{-v, 0, 0} : Color{0, v, 0});
				// c = getColor(v,-1,1);
				printWithColor("█", c);
			}
			cout << endl;

			auto v = engine.similarWords(vec);
			if (engine.wordExists(line)) {
				double norm = engine.wordNorms[engine.getID(line)];
				cout << "Norm: " << norm << endl;
			}
			cout << "Similar to:";
			for (auto pa : v) {
				cout << ' ' << pa.second << " (" << setprecision(3) << pa.first << ")";
			}
			cout << endl;
			// cout << " (" << v.front().first << " ... " << v.back().first << ")" << endl;
		} else {
			string str1 = line.substr(0, i), str2 = line.substr(i+1);
			vector<float> vec1, vec2;
			if (!eval(engine, str1, &vec1)) continue;
			if (!eval(engine, str2, &vec2)) continue;
			float sum = 0;
			rep(i,0,dim) sum += vec1[i] * vec2[i];
			cout << "Similarity: " << sum << endl;
		}
	}
}
