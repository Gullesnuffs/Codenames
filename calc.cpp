#include <bits/stdc++.h>
using namespace std;

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define rrep(i, a, b) for (int i = (a)-1; i >= int(b); --i)
#define trav(x, v) for (auto &x : v)
#define all(v) (v).begin(), (v).end()
#define what_is(x) cout << #x << " is " << x << endl;

typedef float fl;
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<pii> vpi;

struct SimilarityEngine {
	map<string, vector<fl>> vec;
	int dimension;

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
		this->dimension = dimension;

		const int bufSize = 1 << 16;
		char buf[bufSize];
		string word;
		vector<float> values(dimension);
		vector<fl> valuesd;
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
			vec[word] = move(valuesd);
		}
		cerr << " done!" << endl;
		return true;
	}

	fl similarity(const vector<fl> &v1, const vector<fl> &v2) {
		fl ret = 0;
		int dim = (int)v1.size();
		rep(i, 0, dim) {
			ret += v1[i] * v2[i];
		}
		return ret;
	}

	fl similarity(const string &s1, const string &s2) {
		return similarity(vec.at(s1), vec.at(s2));
	}

	const vector<fl> &getVec(const string &s) {
		return vec.at(s);
	}

	bool wordExists(const string &word) {
		return vec.count(word) > 0;
	}

	vector<pair<fl, string>> similarWords(const vector<fl> &v) {
		vector<pair<fl, string>> ret;
		for (auto it = vec.begin(); it != vec.end(); ++it) {
			ret.push_back(make_pair(-similarity(v, it->second), it->first));
		}
		sort(all(ret));
		vector<pair<fl, string>> res;
		rep(i, 0, 10) {
			res.push_back(make_pair(-ret[i].first, ret[i].second));
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
	engine.load("data.bin");
	for (;;) {
again:
		string line;
		cout << "> ";
		getline(cin, line);
		if (!cin) break;
		if (line.empty()) continue;
		size_t i = 0;
		vector<pair<double, string>> stuff;
		char lastSign = '+';
		for (;;) {
			size_t j = line.find_first_of("+-", i);
			string sub = trim(j == string::npos ? line.substr(i) : line.substr(i, j - i));
			size_t k = sub.find_first_not_of(".0123456789");
			if (k == string::npos) {
				cout << "missing word" << endl;
				goto again;
			}
			string dec = sub.substr(0, k);
			string w = trim(sub.substr(k));
			if (!engine.wordExists(w)) {
				cout << "unknown word " << w << endl;
				goto again;
			}
			double val = parse(dec);
			if (lastSign == '-') val = -val;
			stuff.push_back({val, w});
			if (j == string::npos) break;
			lastSign = line[j];
			i = j+1;
		}

		int dim = engine.dimension;
		vector<fl> vec(dim);
		trav(pa, stuff) {
			vector<fl> vec2 = engine.getVec(pa.second);
			rep(i,0,dim) {
				vec[i] += pa.first * vec2[i];
			}
		}

		fl max = 0, min = 0;
		rep(i,0,dim) {
			max = std::max(max, vec[i]);
			min = std::min(min, vec[i]);
		}
		max = std::max(max, -min);
		if (max == 0) max = 1;
		rep(i,0,dim) {
			fl v = vec[i] / max;
			Color c = (v < 0 ? Color{-v, 0, 0} : Color{0, v, 0});
			// c = getColor(v,-1,1);
			printWithColor("█", c);
		}
		cout << endl;

		auto v = engine.similarWords(vec);
		cout << "Similar to:";
		for (auto pa : v) {
			cout << ' ' << pa.second << " (" << setprecision(3) << pa.first << ")";
		}
		cout << endl;
		// cout << " (" << v.front().first << " ... " << v.back().first << ")" << endl;
	}
}
