#include "Bot.h"
#include <cassert>
#include <queue>
#include <map>
#include <iostream>
#include <algorithm>

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define trav(x, v) for (auto &x : v)
#define all(v) (v).begin(), (v).end()

using namespace std;

void Bot::setDifficulty(Difficulty difficulty) {
	if (difficulty == Difficulty::EASY) {
		marginCivilians = 0.08f;
		marginOpponentWords = 0.16f;
		marginAssassins = 0.20f;
		marginOldClue = -0.25f;
		fuzzyWeightAssassin = -0.5f;
		fuzzyWeightOpponent = -0.2f;
		fuzzyWeightMy = 0.1f;
		fuzzyWeightCivilian = -0.1f;
		fuzzyWeightOldClue = -2.0f;
		fuzzyExponent = 15;
		fuzzyOffset = 0.4f;
		minSimilarity = 0.3f;
		multiplierAfterBadWord = 0.5f;
		weightOpponent = -2.0f;
		weightCivilian = -0.4f;
		marginWeight = 0.2f;
		commonWordLimit = 1000;
		commonWordWeight = 0.9f;
		rareWordLimit = 10000;
		rareWordWeight = 0.8f;
		vocabularySize = 30000;
		valueOfOneTurn = 2.0f;
		overlapPenalty = 0.3f;
		desperationFactor[0] = 1.0f;
		desperationFactor[1] = 0.5f;
		desperationFactor[2] = 0.7f;
		desperationFactor[3] = 0.9f;
		singleWordPenalty = -0.3f;
	} else if (difficulty == Difficulty::MEDIUM) {
		marginCivilians = 0.02f;
		marginOpponentWords = 0.04f;
		marginAssassins = 0.07f;
		marginOldClue = -0.25f;
		fuzzyWeightAssassin = -0.2f;
		fuzzyWeightOpponent = -0.1f;
		fuzzyWeightMy = 0.1f;
		fuzzyWeightCivilian = -0.05f;
		fuzzyWeightOldClue = -2.0f;
		fuzzyExponent = 15;
		fuzzyOffset = 0.3f;
		minSimilarity = 0.2f;
		multiplierAfterBadWord = 0.7f;
		weightOpponent = -1.5f;
		weightCivilian = -0.2f;
		marginWeight = 0.1f;
		commonWordLimit = 1000;
		commonWordWeight = 0.9f;
		rareWordLimit = 15000;
		rareWordWeight = 0.8f;
		vocabularySize = 50000;
		valueOfOneTurn = 2.5f;
		overlapPenalty = 0.4f;
		desperationFactor[0] = 1.0f;
		desperationFactor[1] = 0.2f;
		desperationFactor[2] = 0.4f;
		desperationFactor[3] = 0.7f;
		singleWordPenalty = -0.5f;
	} else {
		assert(difficulty == Difficulty::HARD);
		marginCivilians = 0.01f;
		marginOpponentWords = 0.02f;
		marginAssassins = 0.04f;
		marginOldClue = -0.2f;
		fuzzyWeightAssassin = -0.15f;
		fuzzyWeightOpponent = -0.05f;
		fuzzyWeightMy = 0.1f;
		fuzzyWeightCivilian = -0.03f;
		fuzzyWeightOldClue = -1.5f;
		fuzzyExponent = 15;
		fuzzyOffset = 0.25f;
		minSimilarity = 0.15f;
		multiplierAfterBadWord = 0.8f;
		weightOpponent = -1.2f;
		weightCivilian = -0.15f;
		marginWeight = 0.1f;
		commonWordLimit = 1000;
		commonWordWeight = 0.9f;
		rareWordLimit = 20000;
		rareWordWeight = 0.9f;
		vocabularySize = 50000;
		valueOfOneTurn = 3.0f;
		overlapPenalty = 0.5f;
		desperationFactor[0] = 1.0f;
		desperationFactor[1] = 0.1f;
		desperationFactor[2] = 0.3f;
		desperationFactor[3] = 0.5f;
		singleWordPenalty = -0.6f;
	}
}

void Bot::addBoardWord(CardType type, const string &word) {
	boardWords.push_back({type, word, engine.getID(word)});
}

bool Bot::forbiddenWord(const string &word) {
	for (const BoardWord &w : boardWords) {
		if (superOrSubstring(w.word, word))
			return true;
	}
	return false;
}

pair<float, vector<wordID>> Bot::getWordScore(wordID word, vector<ValuationItem> *valuation,
										 bool doInflate) {
	typedef pair<float, BoardWord *> Pa;
	static vector<Pa> v;
	int myWordsLeft = 0, opponentWordsLeft = 0;
	v.clear();

	// Iterate through all words and check how similar the word is to every word on the board.
	// Add some bonuses to account for the colors of the words.
	rep(i, 0, boardWords.size()) {
		float sim = engine.similarity(boardWords[i].id, word);
		if (boardWords[i].type == CardType::CIVILIAN) {
			if (doInflate) {
				sim += marginCivilians;
			}
		} else if (boardWords[i].type == CardType::OPPONENT) {
			if (doInflate) {
				sim += marginOpponentWords;
			}
			opponentWordsLeft++;
		} else if (boardWords[i].type == CardType::ASSASSIN) {
			if (doInflate) {
				sim += marginAssassins;
			}
		} else {
			myWordsLeft++;
		}
		v.push_back({-sim, &boardWords[i]});
	}

	// Sort the similarities to the words on the board
	sort(all(v), [&](const Pa &a, const Pa &b) { return a.first < b.first; });

	// Store the scores for possible use later (e.g show in the client)
	if (valuation != nullptr) {
		valuation->clear();
		trav(it, v) {
			valuation->push_back({-it.first, it.second->word, it.second->type});
		}
	}

	// Compute a fuzzy score
	float baseScore = 0;
	rep(i, 0, v.size()) {
		float weight;
		switch (v[i].second->type) {
			case CardType::MINE:
				weight = fuzzyWeightMy;
				break;
			case CardType::OPPONENT:
				weight = fuzzyWeightOpponent;
				break;
			case CardType::CIVILIAN:
				weight = fuzzyWeightCivilian;
				break;
			case CardType::ASSASSIN:
				weight = fuzzyWeightAssassin;
				break;
			default:
				abort();
		}
		float contribution = weight * sigmoid((-v[i].first - fuzzyOffset) * fuzzyExponent);
		baseScore += contribution;
	}

	// Avoid Bot::clues that are similar to clues the bot has given earlier
	for (auto oldClue : oldClues) {
		float sim = engine.similarity(oldClue, word);
		sim += marginOldClue;
		float contribution = fuzzyWeightOldClue * sigmoid((sim - fuzzyOffset) * fuzzyExponent);
		baseScore += contribution;
	}

	int bestCount = 1;
	float curScore = 0, bestScore = baseScore - 10, lastGood = 0;
	int curCount = 0;
	float mult = 1;

	// Iterate through the words in order and do some scoring...
	// bestCount is the number of words the bot thinks that the rest of the team will manage to
	// guess
	rep(i, 0, v.size()) {
		if (-v[i].first < minSimilarity)
			break;
		CardType type = v[i].second->type;
		if (type == CardType::ASSASSIN)
			break;
		if (type == CardType::OPPONENT) {
			curScore += weightOpponent;
			mult *= multiplierAfterBadWord;
			continue;
		}
		if (type == CardType::MINE) {
			lastGood = -v[i].first;
			curScore += mult * sigmoid((-v[i].first - fuzzyOffset) * fuzzyExponent);
			++curCount;
		}
		if (type == CardType::CIVILIAN) {
			curScore += mult * weightCivilian;
			mult *= multiplierAfterBadWord;
			continue;
		}
		float tmpScore = -1;
		rep(j, i + 1, v.size()) {
			CardType type2 = v[j].second->type;
			if (type2 == CardType::ASSASSIN || type2 == CardType::OPPONENT) {
				tmpScore =
					mult * marginWeight * sigmoid((lastGood - (-v[j].first)) * fuzzyExponent);
				break;
			}
		}
		tmpScore += baseScore + curScore;
		if (curCount == 1) {
			tmpScore += singleWordPenalty;
		}
		if (curCount < myWordsLeft - 1 && opponentWordsLeft <= 3) {
			// Apply penalty because we can't win this turn and the opponent will probably win
			// next turn
			tmpScore *= desperationFactor[opponentWordsLeft];
		}
		if (tmpScore > bestScore) {
			bestScore = tmpScore;
			bestCount = curCount;
		}
	}

	// Create a list of all our words on the board that the bot thinks that the rest of the team
	// will guess
	vector<wordID> targetWords;
	rep(i, 0, v.size()) {
		if ((int)targetWords.size() >= bestCount)
			break;
		CardType type = v[i].second->type;
		if (type == CardType::MINE) {
			targetWords.push_back(v[i].second->id);
		}
	}

	int popularity = engine.getPopularity(word);
	if (popularity < commonWordLimit)
		bestScore *= commonWordWeight;
	else if (popularity > rareWordLimit)
		bestScore *= rareWordWeight;

	bool isInappropriate = inappropriateEngine.isInappropriate(engine.getWord(word));
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

	return make_pair(bestScore, targetWords);
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

vector<Bot::Result> Bot::findBestWords(int count) {
	vector<wordID> candidates = engine.getCommonWords(vocabularySize);
	priority_queue<pair<pair<float, int>, wordID>> pq;
	map<int, int> bitRepresentation;
	int myWordsFound = 0;
	rep(i, 0, boardWords.size()) {
		if (boardWords[i].type == CardType::MINE &&
			!hasInfoAbout.count(engine.getWord(boardWords[i].id))) {
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
			if (bits && newScore > bestScore[bits] &&
				!forbiddenWord(engine.getWord(candidate))) {
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
			res.push_back(Bot::Result{engine.getWord(word), number, score, val});
		}
		sort(all(res));
		return res;
	}

	// Extract the top 'count' words that are not forbidden by the rules
	while ((int)res.size() < count && !pq.empty()) {
		auto pa = pq.top();
		pq.pop();
		if (!forbiddenWord(engine.getWord(pa.second))) {
			float score = pa.first.first;
			int number = -pa.first.second;
			wordID word = pa.second;
			vector<ValuationItem> val;
			getWordScore(word, &val, false);
			res.push_back(Bot::Result{engine.getWord(word), number, score, val});
		}
	}

	return res;
}

void Bot::setHasInfo(string word) {
	hasInfoAbout.insert(word);
}

void Bot::addOldClue(string clue) {
	if (engine.wordExists(clue)) {
		auto wordID = engine.getID(clue);
		oldClues.push_back(wordID);
	}
}
