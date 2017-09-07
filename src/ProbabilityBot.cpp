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
	int myWordsLeft = 0, opponentWordsLeft = 0;

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

		float weight = exp(3*sim);
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
		score[i] = engine.similarity(boardWords[i].id, word) - 0.15;
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
			double p = (2-score[j]) * score[j] + 0.05;
			remaining[j] = p >= distribution01(generator);
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
		for (auto word : boardWords) {
			result.valuations.push_back({ engine.similarity(word.id, item.second), word.word, word.type });
		}
		sort(result.valuations.rbegin(), result.valuations.rend());
		results.push_back(result);

		//cout << item.first.first << " " << item.first.second << " " << dict.getWord(item.second) << endl;
	}

	return results;
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
