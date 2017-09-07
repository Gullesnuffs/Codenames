#pragma once

#include "FuzzyBot.h"
#include "ProbabilityBot.h"
#include "Bot.h"
#include "Dictionary.h"
#include "InappropriateEngine.h"
#include "SimilarityEngine.h"

#include <set>
#include <string>
#include <vector>
#include <memory>

class GameInterface {
	typedef Bot::ValuationItem ValuationItem;
	typedef Bot::Result Result;
	typedef Bot::CardType CardType;
	Dictionary &dict;
	SimilarityEngine &engine;
	std::unique_ptr<Bot> bot;
	std::vector<std::string> myWords, opponentWords, civilianWords, assassinWords;
	std::string myColor;

	void printValuation(const std::string &word, const std::vector<Bot::ValuationItem> &valuation);

	void commandReset();

	void commandSuggestWord();

	void commandHelp();

	void commandBoard();

	void commandModifyBoard(const std::string &command);

	void commandScore();

	std::string inputColor();

   public:
	GameInterface(Dictionary &dict, SimilarityEngine &engine,
				  InappropriateEngine &inappropriateEngine)
		: dict(dict), engine(engine) {
			bot = std::unique_ptr<FuzzyBot>(new FuzzyBot(dict, engine, inappropriateEngine));
		}

	void run();
};
