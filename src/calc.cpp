#include "Dictionary.h"
#include "SimilarityEngine.h"
#include "Word2VecSimilarityEngine.h"
#include "Utilities.h"

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

bool eval(Word2VecSimilarityEngine &engine, Dictionary &dict, const string &line, vector<float> *out) {
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

	int dim = engine.dimension();
	vector<float> vec(dim);
	trav(pa, stuff) {
		vector<float> vec2 = engine.getVector(dict.getID(pa.second));
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
	Dictionary dict;
	Word2VecSimilarityEngine engine(dict);
	engine.load("data.bin", false);
	const int dim = engine.dimension();
	for (;;) {
		string line;
		cout << "> ";
		getline(cin, line);
		if (!cin) break;
		if (line.empty()) continue;
		vector<float> vec;
		size_t i = line.find('*');
		if (i == string::npos) {
			if (!eval(engine, dict, line, &vec)) continue;

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
				double norm = engine.getNorm(dict.getID(line));
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
			if (!eval(engine, dict, str1, &vec1)) continue;
			if (!eval(engine, dict, str2, &vec2)) continue;
			float sum = 0;
			rep(i,0,dim) sum += vec1[i] * vec2[i];
			cout << "Similarity: " << sum << endl;
		}
	}
}
