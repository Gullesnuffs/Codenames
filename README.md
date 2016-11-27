# Codenames
A codenames bot playing the part of a spymaster.

To use this program, you need to do the following:

1. Download the C++ source.

2. Compile it using (for instance) `g++ -std=c++11 -Ofast codenames.cpp`

3. Download a word2vec-like model and save it to `data.txt`. You can download such a model from http://nlp.stanford.edu/projects/glove/. I have used the file glove.6B.zip (300-dimensional vectors), but you would probably get better results with a larger model.

4. Compile and run `preprocess.cpp` using `g++ -std=c++11 -O2 preprocess.cpp -o preprocess && ./preprocess` (will take around 10 seconds to run). This converts `data.txt` into an quicker-to-read binary file `data.bin`.

5. Run the program!
