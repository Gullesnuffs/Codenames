import tensorflow as tf
import numpy as np
import sys

if len(sys.argv) < 3 or len(sys.argv) > 4:
  print("Usage: python3 training.py train-file test-file [result-file]")
  sys.exit()

# Parse samples
trainSamples = np.loadtxt(sys.argv[1], ndmin=2)

testSamples = np.loadtxt(sys.argv[2], ndmin=2)

dim = trainSamples.shape[1]//2

trainSamples1 = trainSamples[:, 0:dim]
trainConceptnet1 = trainSamples1[:, 0:1]
trainGlove1 = trainSamples1[:, 2:3]
trainGloveNorm1 = trainSamples1[:, 3:4]
trainWikisaurus1 = trainSamples1[:, 4:5]

trainSamples2 = trainSamples[:, dim:(dim*2)]
trainConceptnet2 = trainSamples2[:, 0:1]
trainGlove2 = trainSamples2[:, 2:3]
trainGloveNorm2 = trainSamples2[:, 3:4]
trainWikisaurus2 = trainSamples2[:, 4:5]

trainClueGloveNorm = trainSamples1[:, 5:6]

testSamples1 = testSamples[:, 0:dim]
testConceptnet1 = testSamples1[:, 0:1]
testGlove1 = testSamples1[:, 2:3]
testGloveNorm1 = testSamples1[:, 3:4]
testWikisaurus1 = testSamples1[:, 4:5]

testSamples2 = testSamples[:, dim:(dim*2)]
testConceptnet2 = testSamples2[:, 0:1]
testGlove2 = testSamples2[:, 2:3]
testGloveNorm2 = testSamples2[:, 3:4]
testWikisaurus2 = testSamples2[:, 4:5]

testClueGloveNorm = testSamples1[:, 5:6]

conceptnet1 = tf.placeholder(tf.float32, [None, 1])
conceptnet2 = tf.placeholder(tf.float32, [None, 1])
glove1 = tf.placeholder(tf.float32, [None, 1])
glove2 = tf.placeholder(tf.float32, [None, 1])
gloveNorm1 = tf.placeholder(tf.float32, [None, 1])
gloveNorm2 = tf.placeholder(tf.float32, [None, 1])
wikisaurus1 = tf.placeholder(tf.float32, [None, 1])
wikisaurus2 = tf.placeholder(tf.float32, [None, 1])
clueGloveNorm = tf.placeholder(tf.float32, [None, 1])

wikisaurusWeight = tf.Variable(0.00)
gloveWeight = tf.Variable(0.00)
deviationBase = tf.Variable(0.2)
deviationNormCoeff = tf.Variable(0.0)
deviationClueNormCoeff = tf.Variable(0.0)

score1 = conceptnet1 + wikisaurus1 * wikisaurusWeight + glove1 * gloveWeight
score2 = conceptnet2 + wikisaurus2 * wikisaurusWeight + glove2 * gloveWeight

simDiff = score1 - score2

dev1 = tf.exp(deviationBase + deviationNormCoeff / gloveNorm1 + deviationClueNormCoeff / clueGloveNorm)
dev2 = tf.exp(deviationBase + deviationNormCoeff / gloveNorm2 + deviationClueNormCoeff / clueGloveNorm)

stddev = tf.sqrt(tf.square(dev1)+tf.square(dev2))

y = (tf.erf(simDiff / stddev)+1.0)/2.0

cross_entropy = tf.reduce_mean(-tf.log(y))

train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
  sess.run(train_step, feed_dict={conceptnet1: trainConceptnet1, conceptnet2: trainConceptnet2, glove1: trainGlove1, glove2: trainGlove2, gloveNorm1: trainGloveNorm1, gloveNorm2: trainGloveNorm2, wikisaurus1: trainWikisaurus1, wikisaurus2: trainWikisaurus2, clueGloveNorm: trainClueGloveNorm})
  res = sess.run([cross_entropy, gloveWeight, wikisaurusWeight, deviationBase, deviationNormCoeff, deviationClueNormCoeff], feed_dict={conceptnet1: testConceptnet1, conceptnet2: testConceptnet2, glove1: testGlove1, glove2: testGlove2, gloveNorm1: testGloveNorm1, gloveNorm2: testGloveNorm2, wikisaurus1: testWikisaurus1, wikisaurus2: testWikisaurus2, clueGloveNorm: testClueGloveNorm})
  lastAns = res[0]
  #print(str(lastAns))
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
