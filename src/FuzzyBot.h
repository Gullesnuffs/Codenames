#pragma once

#include "Bot.h"
#include "Dictionary.h"
#include "InappropriateEngine.h"
#include "SimilarityEngine.h"
#include "Utilities.h"

#include <set>
#include <string>
#include <vector>

struct FuzzyBot : Bot {
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

	// A set of strings for which the bot has already provided clues
	std::set<std::string> hasInfoAbout;

	// A list of all clues that have already been given to the team
	std::vector<wordID> oldClues;

	FuzzyBot(Dictionary &dict, SimilarityEngine &engine, InappropriateEngine &inappropriateEngine)
		: Bot(dict, engine, inappropriateEngine) {
		setDifficulty(Difficulty::EASY);
	}

	void setDifficulty(Difficulty difficulty);

	std::pair<float, std::vector<wordID>> getWordScore(wordID word,
													   std::vector<ValuationItem> *valuation,
													   bool doInflate);

	std::vector<Result> findBestWords(int count = 20);

	void setHasInfo(std::string word);

	void addOldClue(std::string clue);
};
