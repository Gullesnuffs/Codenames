#pragma once

#include "Bot.h"
#include "Dictionary.h"
#include "InappropriateEngine.h"
#include "SimilarityEngine.h"
#include "Utilities.h"

#include <set>
#include <string>
#include <vector>

struct ProbabilityBot : Bot {
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

	// A set of strings for which the bot has already provided clues
	std::set<std::string> hasInfoAbout;

	// A list of all clues that have already been given to the team
	std::vector<wordID> oldClues;

	ProbabilityBot(Dictionary &dict, SimilarityEngine &engine, InappropriateEngine &inappropriateEngine)
		: Bot(dict, engine, inappropriateEngine) {
		setDifficulty(Difficulty::EASY);
	}

	void setDifficulty(Difficulty difficulty);

	float getWordScore(wordID word);

	float getProbabilityScore(wordID word, int number);

	std::vector<Result> findBestWords(int count = 20);

	void setHasInfo(std::string word);

	void addOldClue(std::string clue);
};
