#include "Bot.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <queue>

using namespace std;

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define trav(x, v) for (auto &x : v)
#define all(v) (v).begin(), (v).end()

void Bot::addBoardWord(CardType type, const string &word) {
	boardWords.push_back({type, word, dict.getID(word)});
}

bool Bot::forbiddenWord(const string &word) {
	for (const BoardWord &w : boardWords) {
		if (superOrSubstring(w.word, word))
			return true;
	}
	return false;
}

void Bot::setWords(const vector<string> &_myWords, const vector<string> &_opponentWords,
				   const vector<string> &_civilianWords, const vector<string> &_assassinWords) {
	myWords = _myWords;
	opponentWords = _opponentWords;
	civilianWords = _civilianWords;
	assassinWords = _assassinWords;
	createBoardWords();
}

void Bot::createBoardWords() {
	boardWords.clear();
	trav(w, myWords) addBoardWord(CardType::MINE, w);
	trav(w, opponentWords) addBoardWord(CardType::OPPONENT, w);
	trav(w, civilianWords) addBoardWord(CardType::CIVILIAN, w);
	trav(w, assassinWords) addBoardWord(CardType::ASSASSIN, w);
}
