#include "Utilities.h"
#include <cmath>

using namespace std;

float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

void eraseFromVector(string word, vector<string> &v) {
	for (int i = 0; i < (int)v.size(); i++) {
		if (v[i] == word) {
			v.erase(v.begin() + i);
			i--;
		}
	}
}
