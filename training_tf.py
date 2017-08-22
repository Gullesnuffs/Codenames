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
trainWikisaurus1 = trainSamples1[:, 4:5]

trainSamples2 = trainSamples[:, dim:(dim*2)]
trainConceptnet2 = trainSamples2[:, 0:1]
trainWikisaurus2 = trainSamples2[:, 4:5]

testSamples1 = testSamples[:, 0:dim]
testConceptnet1 = testSamples1[:, 0:1]
testWikisaurus1 = testSamples1[:, 4:5]

testSamples2 = testSamples[:, dim:(dim*2)]
testConceptnet2 = testSamples2[:, 0:1]
testWikisaurus2 = testSamples2[:, 4:5]

conceptnet1 = tf.placeholder(tf.float32, [None, 1])
conceptnet2 = tf.placeholder(tf.float32, [None, 1])
wikisaurus1 = tf.placeholder(tf.float32, [None, 1])
wikisaurus2 = tf.placeholder(tf.float32, [None, 1])

wikisaurusWeight = tf.Variable(0.00)

score1 = conceptnet1 + wikisaurus1 * wikisaurusWeight
score2 = conceptnet2 + wikisaurus2 * wikisaurusWeight

simDiff = score1 - score2

invstddev = tf.Variable(0.0)

y = (tf.erf(simDiff * invstddev)+1.0)/2.0

cross_entropy = tf.reduce_mean(-tf.log(y))

train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
  sess.run(train_step, feed_dict={conceptnet1: trainConceptnet1, conceptnet2: trainConceptnet2, wikisaurus1: trainWikisaurus1, wikisaurus2: trainWikisaurus2})
  res = sess.run([cross_entropy, invstddev, wikisaurusWeight], feed_dict={conceptnet1: testConceptnet1, conceptnet2: testConceptnet2, wikisaurus1: testWikisaurus1, wikisaurus2: testWikisaurus2})
  lastAns = res[0]
  stddev = 1.0/res[1]
  out_wikisaurusWeight = res[2]

print("Wikisaurus weight: " + str(out_wikisaurusWeight))
print("Standard deviation: " + str(stddev/2))

if len(sys.argv) >= 4:
  f = open(sys.argv[3], 'w')
  f.write(str(lastAns))
  f.close()
  sys.exit()
