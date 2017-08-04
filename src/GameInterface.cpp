#include "GameInterface.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include "Utilities.h"

#define rep(i, a, b) for (int i = (a); i < int(b); ++i)
#define rrep(i, a, b) for (int i = (a)-1; i >= int(b); --i)
#define trav(x, v) for (auto &x : v)
#define all(v) (v).begin(), (v).end()

using namespace std;

/** Returns suffix on number such as 'th' for 5 or 'nd' for 2 */
string orderSuffix(int p) {
	if (p % 10 == 1 && p % 100 != 11) {
		return "st";
	} else if (p % 10 == 2 && p % 100 != 12) {
		return "nd";
	} else if (p % 10 == 3 && p % 100 != 13) {
		return "rd";
	} else {
		return "th";
	}
}

void GameInterface::printValuation(const string &word,
								   const vector<Bot::ValuationItem> &valuation) {
	cout << "Printing statistics for \"" << denormalize(word) << "\"" << endl;
	map<CardType, string> desc;
	desc[CardType::MINE] = "(My)";
	desc[CardType::OPPONENT] = "(Opponent)";
	desc[CardType::CIVILIAN] = "(Civilian)";
	desc[CardType::ASSASSIN] = "(Assassin)";
	trav(item, valuation) {
		cout << setprecision(6) << fixed << item.score << "\t";
		cout << denormalize(item.word) << " " << desc[item.type] << endl;
	}
	cout << endl;
}

void GameInterface::commandReset() {
	myWords.clear();
	opponentWords.clear();
	civilianWords.clear();
	assassinWords.clear();
	bot.setWords(myWords, opponentWords, civilianWords, assassinWords);
}

void GameInterface::commandSuggestWord() {
	cout << "Thinking..." << endl;
	vector<Result> results = bot.findBestWords();
	if (results.empty()) {
		cout << "Not a clue." << endl;
	} else {
		Result &best = results[0];
		printValuation(best.word, best.valuations);

		// Print a list with the best clues
		rep(i, 0, (int)results.size()) {
			auto res = results[i];
			cout << (i + 1) << "\t" << setprecision(3) << fixed << res.score << "\t"
				 << engine.stat(dict.getID(res.word)) << "\t" << res.word << " " << res.number
				 << endl;
		}
		cout << endl;

		int p = dict.getPopularity(dict.getID(best.word));
		cout << "The best clue found is " << denormalize(best.word) << " " << best.number << endl;
		cout << best.word << " is the " << p << orderSuffix(p) << " most popular word" << endl;
	}
}

void GameInterface::commandHelp() {
	cout << "The following commands are available:" << endl << endl;
	cout << "r <word>\t-\tAdd a red spy to the board" << endl;
	cout << "b <word>\t-\tAdd a blue spy to the board" << endl;
	cout << "c <word>\t-\tAdd a civilian to the board" << endl;
	cout << "a <word>\t-\tAdd an assassin to the board" << endl;
	cout << "- <word>\t-\tRemove a word from the board" << endl;
	cout << "go\t\t-\tReceive clues" << endl;
	cout << "reset\t\t-\tClear the board" << endl;
	cout << "board\t\t-\tPrints the words currently on the board" << endl;
	cout << "score <word>\t-\tCompute how good a given clue would be" << endl;
	cout << "quit\t\t-\tTerminates the program" << endl;
}

void GameInterface::commandBoard() {
	cout << "My spies:";
	for (auto word : myWords) {
		cout << " " << denormalize(word);
	}
	cout << endl;
	cout << "Opponent spies:";
	for (auto word : opponentWords) {
		cout << " " << denormalize(word);
	}
	cout << endl;
	cout << "Civilians:";
	for (auto word : civilianWords) {
		cout << " " << denormalize(word);
	}
	cout << endl;
	cout << "Assassins:";
	for (auto word : assassinWords) {
		cout << " " << denormalize(word);
	}
	cout << endl;
}

void GameInterface::commandModifyBoard(const string &command) {
	vector<string> *v = NULL;
	if (command == myColor) {
		v = &myWords;
	} else if (command == "b" || command == "r") {
		v = &opponentWords;
	} else if (command == "g" || command == "c") {
		v = &civilianWords;
	} else if (command == "a") {
		v = &assassinWords;
	} else if (command == "-") {
		string word;
		cin >> word;
		word = normalize(word);
		eraseFromVector(word, myWords);
		eraseFromVector(word, opponentWords);
		eraseFromVector(word, civilianWords);
		eraseFromVector(word, assassinWords);
	}

	if (v != NULL) {
		string word;
		cin >> word;
		word = normalize(word);
		if (engine.wordExists(word)) {
			v->push_back(word);
		} else {
			cout << denormalize(word) << " was not found in the dictionary" << endl;
		}
	}
	bot.setWords(myWords, opponentWords, civilianWords, assassinWords);
}

void GameInterface::commandScore() {
	string word;
	cin >> word;
	if (!engine.wordExists(word)) {
		cout << denormalize(word) << " was not found in the dictionary" << endl;
		return;
	}
	vector<ValuationItem> val;
	pair<float, vector<wordID>> res = bot.getWordScore(dict.getID(word), &val, true);
	printValuation(word, val);
	cout << denormalize(word) << " " << res.second.size() << " has score " << res.first << endl;
}

string GameInterface::inputColor() {
	while (true) {
		string color;
		cin >> color;
		color = normalize(color);
		if (color == "b" || color == "blue") {
			return "b";
		}
		if (color == "r" || color == "red") {
			return "r";
		}
	}
}

void GameInterface::run() {
	cout << "Type \"help\" for help" << endl;
	cout << "My color (b/r): ";
	myColor = inputColor();

	while (true) {
		string command;
		cin >> command;
		if (!cin)
			break;
		command = normalize(command);

		if (command.size() == 1 && string("rgbac-").find(command) != string::npos) {
			commandModifyBoard(command);
		} else if (command == "play" || command == "go") {
			commandSuggestWord();
		} else if (command == "quit" || command == "exit") {
			break;
		} else if (command == "reset") {
			commandReset();
		} else if (command == "help" || command == "\"help\"") {
			commandHelp();
		} else if (command == "board") {
			commandBoard();
		} else if (command == "score") {
			commandScore();
		} else {
			cout << "Unknown command \"" << command << "\"" << endl;
		}
	}
}