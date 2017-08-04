#pragma once

#include<unordered_set>
#include <string>

enum InappropriateMode {
	BlockInappropriate,
	AllowInappropriate,
	BoostInappropriate,
};

struct InappropriateEngine {
	std::unordered_set<std::string> inappropriateWords;

	void load(std::string fileName);
	bool isInappropriate(std::string word);
};