#pragma once

#include <string>

/** Represents a single word or phrase in a similarity engine */
enum wordID : int {};

std::string normalize(std::string s);

std::string denormalize(std::string s);

/** True if a is a super or substring of b or vice versa */
bool superOrSubstring(const std::string &a, const std::string &b);