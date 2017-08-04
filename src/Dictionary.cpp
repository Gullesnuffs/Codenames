#include "Dictionary.h"

using namespace std;

string normalize(string s) {
	for (auto &c : s) {
		if ('A' <= c && c <= 'Z') {
			c = (char)tolower(c);
		} else if (c == ' ') {
			c = '_';
		}
	}
	return s;
}

string denormalize(string s) {
	for (auto &c : s) {
		if (c == '_') {
			c = ' ';
		}
	}
	return s;
}

bool superOrSubstring(const string &a, const string &b) {
	auto lowerA = normalize(a);
	auto lowerB = normalize(b);
	return lowerA.find(lowerB) != string::npos || lowerB.find(lowerA) != string::npos;
}