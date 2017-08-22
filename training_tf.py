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
  def __init__(self, clueNorm, wikisaurusWeight, gloveWeight):
    self.conceptnet = tf.placeholder(tf.float32, [None, 1])
    self.glove = tf.placeholder(tf.float32, [None, 1])
    self.gloveNorm = tf.placeholder(tf.float32, [None, 1])
    self.wikisaurus = tf.placeholder(tf.float32, [None, 1])
    self.score = self.conceptnet + self.wikisaurus * wikisaurusWeight + self.glove * gloveWeight


clueGloveNorm = tf.placeholder(tf.float32, [None, 1])

wikisaurusWeight = tf.Variable(0.00)
gloveWeight = tf.Variable(0.00)
deviationBase = tf.Variable(0.2)
deviationNormCoeff = tf.Variable(0.0)
deviationClueNormCoeff = tf.Variable(0.0)

model1 = Model(clueGloveNorm, wikisaurusWeight, gloveWeight)
model2 = Model(clueGloveNorm, wikisaurusWeight, gloveWeight)

simDiff = model1.score - model2.score

dev1 = tf.exp(deviationBase + deviationNormCoeff / model1.gloveNorm + deviationClueNormCoeff / clueGloveNorm)
dev2 = tf.exp(deviationBase + deviationNormCoeff / model2.gloveNorm + deviationClueNormCoeff / clueGloveNorm)

stddev = tf.sqrt(tf.square(dev1) + tf.square(dev2))

y = (tf.erf(simDiff / stddev) + 1.0) / 2.0

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
