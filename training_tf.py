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
    self.conceptnet = data[:, 0:1]
    self.glove = data[:, 2:3]
    self.gloveNorm = data[:, 3:4]
    self.wikisaurus = data[:, 4:5]
    self.clueGloveNorm = data[:, 5:6]


train1 = Feature(trainSamples[:, 0:dim])
train2 = Feature(trainSamples[:, dim:(dim * 2)])
test1 = Feature(testSamples[:, 0:dim])
test2 = Feature(testSamples[:, dim:(dim * 2)])

clueGloveNorm = tf.placeholder(tf.float32, [None, 1])


class Model:
  def __init__(self, clueNorm, wikisaurusWeight, gloveWeight, deviationBase, deviationNormCoeff):
    self.conceptnet = tf.placeholder(tf.float32, [None, 1])
    self.glove = tf.placeholder(tf.float32, [None, 1])
    self.gloveNorm = tf.placeholder(tf.float32, [None, 1])
    self.wikisaurus = tf.placeholder(tf.float32, [None, 1])
    self.score = self.conceptnet + self.wikisaurus * wikisaurusWeight + self.glove * gloveWeight
    # Standard deviation for the score
    self.dev = tf.exp(deviationBase + deviationNormCoeff / self.gloveNorm + deviationClueNormCoeff / clueNorm)


clueGloveNorm = tf.placeholder(tf.float32, [None, 1])

wikisaurusWeight = tf.Variable(0.00)
gloveWeight = tf.Variable(0.00)
deviationBase = tf.Variable(0.2)
deviationNormCoeff = tf.Variable(0.0)
deviationClueNormCoeff = tf.Variable(0.0)

model1 = Model(clueGloveNorm, wikisaurusWeight, gloveWeight, deviationBase, deviationNormCoeff)
model2 = Model(clueGloveNorm, wikisaurusWeight, gloveWeight, deviationBase, deviationNormCoeff)

scoreDiff = model1.score - model2.score

# Standard deviation fof the difference of the scores (scoreDiff).
# If the standard deviation would be zero we would always pick the word with the highest score.
stddev = tf.sqrt(tf.square(model1.dev) + tf.square(model2.dev))

# Calculate the probability that we will pick the first word before the second word.
# In the test data the first word is the "correct" word (i.e the one a human picked first).
# We want to maximize this value
y = (tf.erf(scoreDiff / stddev) + 1.0) / 2.0

cross_entropy = tf.reduce_mean(-tf.log(y))

train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()


def constructFeedDict(model1, feature1, model2, feature2):
  return {model1.conceptnet: feature1.conceptnet,
          model2.conceptnet: feature2.conceptnet,
          model1.glove: feature1.glove,
          model2.glove: feature2.glove,
          model1.gloveNorm: feature1.gloveNorm,
          model2.gloveNorm: feature2.gloveNorm,
          model1.wikisaurus: feature1.wikisaurus,
          model2.wikisaurus: feature2.wikisaurus,
          clueGloveNorm: feature1.clueGloveNorm
          }


for _ in range(1000):
  sess.run(train_step, feed_dict=constructFeedDict(model1, train1, model2, train2))
  res = sess.run([cross_entropy, gloveWeight, wikisaurusWeight, deviationBase, deviationNormCoeff, deviationClueNormCoeff], feed_dict=constructFeedDict(model1, test1, model2, test2))
  lastAns = res[0]
  # print(str(lastAns))
  out_gloveWeight = res[1]
  out_wikisaurusWeight = res[2]
  out_deviationBase = res[3]
  out_deviationNormCoeff = res[4]
  out_deviationClueNormCoeff = res[5]

print("Glove weight = " + str(out_gloveWeight))
print("Wikisaurus weight = " + str(out_wikisaurusWeight))
print("deviationBase = " + str(out_deviationBase))
print("deviationNormCoeff = " + str(out_deviationNormCoeff))
print("deviationClueNormCoeff = " + str(out_deviationClueNormCoeff))

if len(sys.argv) >= 4:
  f = open(sys.argv[3], 'w')
  f.write(str(lastAns))
  f.close()
  sys.exit()
