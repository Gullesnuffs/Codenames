#pragma once

#include "Dictionary.h"
#include "InappropriateEngine.h"
#include "SimilarityEngine.h"
#include "Utilities.h"

#include <set>
#include <string>
#include <vector>

struct Bot {
	enum class CardType { MINE, OPPONENT, CIVILIAN, ASSASSIN };
	enum class Difficulty { EASY, MEDIUM, HARD };

	struct BoardWord {
		CardType type;
		std::string word;
		wordID id;
	};

	struct ValuationItem {
		float score;
		std::string word;
		CardType type;

		bool operator< (const ValuationItem other) const {
			return score < other.score;
		}
	};
	struct Result {
		std::string word;
		int number;
		float score;
		std::vector<ValuationItem> valuations;

		bool operator<(const Result &other) const {
			return score > other.score;
		}
	};

	InappropriateMode inappropriateMode;

	// Score multiplier for inappropriate words when using the BoostInappropriate mode
	// See inappropriateMode
	float inappropriateBoost = 4;

	Dictionary &dict;
	SimilarityEngine &engine;
	InappropriateEngine &inappropriateEngine;

	std::vector<std::string> myWords, opponentWords, civilianWords, assassinWords;
	std::vector<BoardWord> boardWords;

	Bot(Dictionary &dict, SimilarityEngine &engine, InappropriateEngine &inappropriateEngine)
		: dict(dict), engine(engine), inappropriateEngine(inappropriateEngine) {}

 	virtual ~Bot() {}

	virtual void setDifficulty(Difficulty difficulty) = 0;

	void addBoardWord(CardType type, const std::string &word);

	bool forbiddenWord(const std::string &word);

	void setWords(const std::vector<std::string> &_myWords,
				  const std::vector<std::string> &_opponentWords,
				  const std::vector<std::string> &_civilianWords,
				  const std::vector<std::string> &_assassinWords);

	void createBoardWords();

	virtual std::vector<Result> findBestWords(int count = 20) = 0;

	virtual void setHasInfo(std::string word) = 0;

	virtual void addOldClue(std::string clue) = 0;
};
