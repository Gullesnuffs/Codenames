#pragma once

#include <set>
#include <string>
#include <vector>
#include "Bot.h"
#include "Dictionary.h"
#include "InappropriateEngine.h"
#include "SimilarityEngine.h"

class GameInterface {
	typedef Bot::ValuationItem ValuationItem;
	typedef Bot::Result Result;
	typedef Bot::CardType CardType;
	Dictionary &dict;
	SimilarityEngine &engine;
	Bot bot;
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
		: dict(dict), engine(engine), bot(dict, engine, inappropriateEngine) {}

	void run();
};