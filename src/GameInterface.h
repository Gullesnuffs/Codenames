#pragma once

#include "Bot.h"
#include "SimilarityEngine.h"
#include "Dictionary.h"
#include "InappropriateEngine.h"
#include <vector>
#include <string>
#include <set>

class GameInterface {
	typedef Bot::ValuationItem ValuationItem;
	typedef Bot::Result Result;
	typedef Bot::CardType CardType;
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
	GameInterface(SimilarityEngine &engine, InappropriateEngine &inappropriateEngine)
		: engine(engine), bot(engine, inappropriateEngine) {}

	void run();
};