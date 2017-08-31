import tensorflow as tf
import numpy as np
import sys

if len(sys.argv) < 3 or len(sys.argv) > 4:
  print("Usage: python3 training.py train-file test-file [result-file]")
  sys.exit()

# Parse samples
trainSamples = np.loadtxt(sys.argv[1], ndmin=2)
testSamples = np.loadtxt(sys.argv[2], ndmin=2)

dim = trainSamples.shape[1] // 2


class Feature:
  def __init__(self, data):
    self.conceptnetNorm = data[:, 0:1]
    self.gloveNorm = data[:, 1:2]
    self.clueGloveNorm = data[:, 2:3]
    self.similarities = data[:, 3:data.shape[1]]


train1 = Feature(trainSamples[:, 0:dim])
train2 = Feature(trainSamples[:, dim:(dim * 2)])
test1 = Feature(testSamples[:, 0:dim])
test2 = Feature(testSamples[:, dim:(dim * 2)])
numSimilarities = train1.similarities.shape[1]

clueGloveNorm = tf.placeholder(tf.float32, [None, 1])


class Model:
  def __init__(self, clueNorm, similarityWeights, deviationBase, deviationNormCoeff):
    self.similarities = tf.placeholder(tf.float32, [None, similarityWeights.shape[0]])
    self.gloveNorm = tf.placeholder(tf.float32, [None, 1])
    # This is essentially a dot product between the similarities and the similarity weights
    # Each similarity should be multiplied by its corresponding weight
    self.score = tf.reduce_sum(tf.multiply(self.similarities, similarityWeights), axis=1, keep_dims=True)
    # Standard deviation for the score
    self.dev = tf.exp(deviationBase + deviationNormCoeff / self.gloveNorm + deviationClueNormCoeff / clueNorm)


clueGloveNorm = tf.placeholder(tf.float32, [None, 1])

similarityWeights = tf.get_variable("similarity_weights", [numSimilarities], dtype=tf.float32)
deviationBase = tf.Variable(0.2)
deviationNormCoeff = tf.Variable(0.0)
deviationClueNormCoeff = tf.Variable(0.0)

model1 = Model(clueGloveNorm, similarityWeights, deviationBase, deviationNormCoeff)
model2 = Model(clueGloveNorm, similarityWeights, deviationBase, deviationNormCoeff)

scoreDiff = model1.score - model2.score

# Standard deviation fof the difference of the scores (scoreDiff).
# If the standard deviation would be zero we would always pick the word with the highest score.
stddev = tf.sqrt(tf.square(model1.dev) + tf.square(model2.dev))

# Calculate the probability that we will pick the first word before the second word.
# In the test data the first word is the "correct" word (i.e the one a human picked first).
# We want to maximize this value
y = (tf.erf(scoreDiff / stddev) + 1.0) / 2.0

cross_entropy = tf.reduce_mean(-tf.log(y))

# Keep the first similarity weight at a constant 1
loss = cross_entropy + tf.square(1 - similarityWeights[0])

train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()


def constructFeedDict(model1, feature1, model2, feature2):
  return {model1.similarities: feature1.similarities,
          model2.similarities: feature2.similarities,
          model1.gloveNorm: feature1.gloveNorm,
          model2.gloveNorm: feature2.gloveNorm,
          clueGloveNorm: feature1.clueGloveNorm
          }


for i in range(500):
  _, ce = sess.run([train_step, cross_entropy], feed_dict=constructFeedDict(model1, train1, model2, train2))
  sys.stdout.write("\r" + str(i).ljust(8) + str(ce))

res = sess.run([cross_entropy, similarityWeights, deviationBase, deviationNormCoeff, deviationClueNormCoeff], feed_dict=constructFeedDict(model1, test1, model2, test2))
lastAns = res[0]
# print(str(lastAns))
out_similarityWeights = res[1]
out_deviationBase = res[2]
out_deviationNormCoeff = res[3]
out_deviationClueNormCoeff = res[4]

print()
print("Similarity Weights = " + "  ".join(["{0:.2f}".format(x) for x in out_similarityWeights]))
print("deviationBase = " + str(out_deviationBase))
print("deviationNormCoeff = " + str(out_deviationNormCoeff))
print("deviationClueNormCoeff = " + str(out_deviationClueNormCoeff))
print("cross entropy = " + str(lastAns))

if len(sys.argv) >= 4:
  f = open(sys.argv[3], 'w')
  f.write("{0:.10f}".format(lastAns))
  f.close()
  sys.exit()
