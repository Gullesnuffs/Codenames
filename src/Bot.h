#pragma once

#include "Dictionary.h"
#include "Utilities.h"
#include "SimilarityEngine.h"
#include "InappropriateEngine.h"

#include <string>
#include <vector>
#include <set>

struct Bot {
	enum class CardType { MINE, OPPONENT, CIVILIAN, ASSASSIN };
	enum class Difficulty { EASY, MEDIUM, HARD };

	// Give a similarity bonus to "bad" words
	float marginCivilians;
	float marginOpponentWords;
	float marginAssassins;
	float marginOldClue;

	// Constants used in scoring function based
	// on the sigmoid function of the similarities
	float fuzzyWeightAssassin;
	float fuzzyWeightOpponent;
	float fuzzyWeightMy;
	float fuzzyWeightCivilian;
	float fuzzyWeightOldClue;
	float fuzzyExponent;
	float fuzzyOffset;

	// Assume that we will never succeed if the similarity
	// is at most minSimilarity
	float minSimilarity;

	// Good words with smaller similarity than civilians and opponent
	// spies are worth less
	float multiplierAfterBadWord;

	// How bad is it if there is an opponent word with high similarity
	float weightOpponent;

	// How bad is it if there is a civilian word with high similarity
	float weightCivilian;

	// How important is it that the last good word has greater
	// similarity than the next bad word
	float marginWeight;

	// Number of words that are considered common
	int commonWordLimit;

	// Avoid common words
	float commonWordWeight;

	// Number of words that are not considered rare
	int rareWordLimit;

	// Avoid rare words
	float rareWordWeight;

	// Consider only the 50000 most common words
	int vocabularySize;

	// An approximation of the number of correct words we expect each turn
	float valueOfOneTurn;

	float overlapPenalty;

	// Apply penalties to clues with small numbers based on the number of
	// remaining opponent words
	float desperationFactor[4];

	// Apply a penalty to words that only cover a single word
	float singleWordPenalty;

	InappropriateMode inappropriateMode;

	// Score multiplier for inappropriate words when using the BoostInappropriate mode
	// See inappropriateMode
	float inappropriateBoost = 4;

	// A set of strings for which the bot has already provided clues
	std::set<std::string> hasInfoAbout;

	// A list of all clues that have already been given to the team
	std::vector<wordID> oldClues;

	SimilarityEngine &engine;
	InappropriateEngine &inappropriateEngine;

	void setDifficulty(Difficulty difficulty);

	Bot(SimilarityEngine &engine, InappropriateEngine &inappropriateEngine)
		: engine(engine), inappropriateEngine(inappropriateEngine) {
		setDifficulty(Difficulty::EASY);
	}

	std::vector<std::string> myWords, opponentWords, civilianWords, assassinWords;
	struct BoardWord {
		CardType type;
		std::string word;
		wordID id;
	};
	std::vector<BoardWord> boardWords;
	void addBoardWord(CardType type, const std::string &word);

	bool forbiddenWord(const std::string &word);

	struct ValuationItem {
		float score;
		std::string word;
		CardType type;
	};

	std::pair<float, std::vector<wordID>> getWordScore(wordID word, std::vector<ValuationItem> *valuation,
											 bool doInflate);

	void setWords(const std::vector<std::string> &_myWords, const std::vector<std::string> &_opponentWords,
				  const std::vector<std::string> &_civilianWords, const std::vector<std::string> &_assassinWords);

	void createBoardWords();

	struct Result {
		std::string word;
		int number;
		float score;
		std::vector<ValuationItem> valuations;

		bool operator<(const Result &other) const {
			return score > other.score;
		}
	};

	std::vector<Result> findBestWords(int count = 20);

	void setHasInfo(std::string word);

	void addOldClue(std::string clue);
};