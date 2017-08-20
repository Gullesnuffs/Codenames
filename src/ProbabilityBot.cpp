#include "ProbabilityBot.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <queue>
#include <random>

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define trav(x, v) for (auto &x : v)
#define all(v) (v).begin(), (v).end()

using namespace std;

void ProbabilityBot::setDifficulty(Difficulty difficulty) {
	vocabularySize = 30000;
}

float ProbabilityBot::getWordScore(wordID word) {
	typedef pair<float, BoardWord *> Pa;
	static vector<Pa> v;
	int myWordsLeft = 0, opponentWordsLeft = 0;
	v.clear();

	// Iterate through all words and check how similar the word is to every word on the board.
	// Add some bonuses to account for the colors of the words.
	float totalWeight = 0;
	float totalScore = 0;
	rep(i, 0, boardWords.size()) {
		float sim = engine.similarity(boardWords[i].id, word);
		float value = 0;
		if (boardWords[i].type == CardType::CIVILIAN) {
			value = 0;
		} else if (boardWords[i].type == CardType::OPPONENT) {
			value = -1;
		} else if (boardWords[i].type == CardType::ASSASSIN) {
			value = -3;
		} else {
			value = 1;
		}

		float weight = exp(5*sim);
		totalWeight += weight;
		totalScore += weight * value;
	}

	totalScore /= totalWeight;
	return totalScore;

	/*bool isInappropriate = inappropriateEngine.isInappropriate(word);
	switch (inappropriateMode) {
		case BlockInappropriate:
			if (isInappropriate) {
				bestScore = -numeric_limits<float>::infinity();
			}
			break;
		case BoostInappropriate:
			if (isInappropriate) {
				bestScore *= inappropriateBoost;
			}
			break;
		case AllowInappropriate:
			break;
	}

	return make_pair(bestScore, targetWords);*/
}

float ProbabilityBot::getProbabilityScore(wordID word, int number) {
	vector<float> score(boardWords.size());
	for(int i = 0; i < boardWords.size(); i++) {
		score[i] = engine.similarity(boardWords[i].id, word);
	}
	vector<float> scoreWithNoise(boardWords.size());
	int simulations = 1000;
	double totScore = 0;
	vector<bool> remaining(boardWords.size());
	normal_distribution<double> distribution(0.0, 0.24 * 0.5);
	uniform_real_distribution<double> distribution01(0.0, 1.0);
	default_random_engine generator;

	for (int i = 0; i < simulations; i++) {
		for (int j = 0; j < boardWords.size(); j++) {
			remaining[j] = score[j] >= distribution01(generator);
			//remaining[j] = true;
		}
		//cerr << "Simulation " << (i+1) << endl;
		for (int j = 0; j < boardWords.size(); j++) {
			scoreWithNoise[j] = score[j] + distribution(generator);
			//cerr << boardWords[j].word << " " << score[j] << " " << scoreWithNoise[j] << endl;
		}
		double simulationScore = 0;
		for (int j = 0; j < number; j++) {
			int chosen = -1;
			for (int k = 0; k < boardWords.size(); k++) {
				if (remaining[k]) {
					if (chosen == -1 || scoreWithNoise[k] > scoreWithNoise[chosen]) {
						chosen = k;
					}
				}
			}

			if (chosen == -1) break;

			//cerr << "Chose " << boardWords[chosen].word << endl;
			remaining[chosen] = false;
			if (boardWords[chosen].type == CardType::CIVILIAN) {
				break;
			}
			if (boardWords[chosen].type == CardType::OPPONENT) {
				simulationScore -= 1;
				break;
			}
			if (boardWords[chosen].type == CardType::ASSASSIN) {
				simulationScore -= 3;
				break;
			}
			if (boardWords[chosen].type == CardType::MINE) {
				simulationScore += 1;
			}
		}
		totScore += simulationScore;
	}
	return totScore / simulations;
}

vector<Bot::Result> ProbabilityBot::findBestWords(int count) {
	vector<wordID> candidates = dict.getCommonWords(vocabularySize);
	priority_queue<pair<float, wordID>> pq;


	for (auto candidate : candidates) {
		pq.push(make_pair(getWordScore(candidate), candidate));
	}


	vector<wordID> subset;
	while (subset.size() < 500 && !pq.empty()) {
		auto item = pq.top();
		if (!forbiddenWord(dict.getWord(item.second))) {
			subset.push_back(item.second);
			//cout << item.first << " " << dict.getWord(item.second) << endl;
		}

		pq.pop();
	}

	vector<pair<pair<float, int>, wordID>> simulationScores;
	for (auto clue : subset) {
		float bestScore = -10000;
		int bestNum = 0;
		for (int num = 1; num <= 9; num++) {
			float score = getProbabilityScore(clue, num);
			if (score > bestScore) {
				bestScore = score;
				bestNum = num;
			}
		}

		simulationScores.push_back(make_pair(make_pair(bestScore, bestNum), clue));
	}

	sort(simulationScores.rbegin(), simulationScores.rend());

	vector<Bot::Result> results;
	for (int i = 0; i < 10; i++) {
		auto item = simulationScores[i];
		Result result;
		result.word = dict.getWord(item.second);
		result.number = item.first.second;
		result.score = item.first.first;
		result.valuations = vector<ValuationItem>(0);
		results.push_back(result);

		//cout << item.first.first << " " << item.first.second << " " << dict.getWord(item.second) << endl;
	}

	return results;

	/*map<int, int> bitRepresentation;
	int myWordsFound = 0;
	rep(i, 0, boardWords.size()) {
		if (boardWords[i].type == CardType::MINE &&
			!hasInfoAbout.count(dict.getWord(boardWords[i].id))) {
			bitRepresentation[boardWords[i].id] = (1 << myWordsFound);
			++myWordsFound;
		} else {
			bitRepresentation[boardWords[i].id] = 0;
		}
	}

	bool usePlanning = (myWordsFound && myWordsFound <= 9);
	vector<int> minMovesNeeded;
	vector<float> bestScore;
	vector<float> bestClueScore;
	vector<wordID> bestWord;
	vector<int> bestParent;
	if (usePlanning) {
		minMovesNeeded = vector<int>((1 << myWordsFound), 1000);
		bestScore = vector<float>((1 << myWordsFound), -1000);
		bestClueScore = vector<float>((1 << myWordsFound), -1000);
		bestWord = vector<wordID>(1 << myWordsFound);
		bestParent = vector<int>(1 << myWordsFound);
		minMovesNeeded[0] = 0;
		bestScore[0] = 0;
	}

	for (wordID candidate : candidates) {
		pair<float, vector<wordID>> res = getWordScore(candidate, nullptr, true);
		pq.push({{res.first, -((int)res.second.size())}, candidate});
		if (res.second.size() > 0 && usePlanning) {
			int bits = 0;
			for (int matchedWord : res.second) {
				bits |= bitRepresentation[matchedWord];
			}
			float newScore = res.first - valueOfOneTurn;
			if (bits && newScore > bestScore[bits] && !forbiddenWord(dict.getWord(candidate))) {
				minMovesNeeded[bits] = 1;
				bestScore[bits] = newScore;
				bestWord[bits] = candidate;
				bestParent[bits] = 0;
			}
		}
	}

	vector<Bot::Result> res;

	if (usePlanning) {
		// Make a plan so that every word is covered by a clue in as few moves as possible

		for (int i = 0; i < (1 << myWordsFound); ++i) {
			if (minMovesNeeded[i] != 1)
				continue;
			for (int j = 0; j < (1 << myWordsFound); ++j) {
				int k = (i | j);
				if (k == i || k == j)
					continue;
				float newScore = bestScore[i] + bestScore[j];
				newScore -= __builtin_popcount(i & j) * overlapPenalty;
				if (newScore > bestScore[k]) {
					minMovesNeeded[k] = minMovesNeeded[j] + 1;
					bestScore[k] = newScore;
					bestClueScore[k] = bestScore[i];
					bestParent[k] = j;
					bestWord[k] = bestWord[i];
				}
			}
		}

		// Reconstruct the best sequence of words
		int bits = (1 << myWordsFound) - 1;
		while (bits != 0) {
			cerr << bits << endl;
			wordID word = bestWord[bits];
			bits = bestParent[bits];
			vector<ValuationItem> val;
			auto wordScore = getWordScore(word, &val, false);
			float score = wordScore.first;
			int number = (int)wordScore.second.size();
			res.push_back(Bot::Result{dict.getWord(word), number, score, val});
		}
		sort(all(res));
		return res;
	}

	// Extract the top 'count' words that are not forbidden by the rules
	while ((int)res.size() < count && !pq.empty()) {
		auto pa = pq.top();
		pq.pop();
		if (!forbiddenWord(dict.getWord(pa.second))) {
			float score = pa.first.first;
			int number = -pa.first.second;
			wordID word = pa.second;
			vector<ValuationItem> val;
			getWordScore(word, &val, false);
			res.push_back(Bot::Result{dict.getWord(word), number, score, val});
		}
	}

	return res;*/
}

void ProbabilityBot::setHasInfo(string word) {
	hasInfoAbout.insert(word);
}

void ProbabilityBot::addOldClue(string clue) {
	if (engine.wordExists(clue)) {
		auto wordID = dict.getID(clue);
		oldClues.push_back(wordID);
	}
}
