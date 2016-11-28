# Codenames
A [codenames](http://czechgames.com/en/codenames/) bot playing the part of a spymaster.

To use this program, you need to do the following:

1. Download the C++ source.

2. Compile it using (for instance) `g++ -std=c++11 -Ofast codenames.cpp -o codenames`

3. Take any binary word2vec-like model from `models/` and copy it to `data.bin`.
   Alternatively, download one in text format from http://nlp.stanford.edu/projects/glove/ (glove.840B.300d works well), and convert it to binary format using `preprocess.cpp`.
   Capping to ~50000 words is reasonable.

4. Run the program!
