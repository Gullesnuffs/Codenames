#pragma once

#include <string>
#include <unordered_set>
#include "Dictionary.h"

enum InappropriateMode {
	BlockInappropriate,
	AllowInappropriate,
	BoostInappropriate,
};

struct InappropriateEngine {
   private:
	std::vector<bool> inappropriateWords;

   public:
	InappropriateEngine(const std::string& filePath, const Dictionary& dict);

	void load(const std::string& filePath, const Dictionary& dict);
	bool isInappropriate(wordID id) const;
};