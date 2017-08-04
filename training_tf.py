import tensorflow as tf
import numpy as np
import sys
import math

if len(sys.argv) < 3 or len(sys.argv) > 4:
  print("Usage: python3 training.py train-file test-file [result-file]")
  sys.exit()

# Parse samples
trainSamples = np.loadtxt(sys.argv[1], ndmin=2)

testSamples = np.loadtxt(sys.argv[2], ndmin=2)

dim = trainSamples.shape[1]//2

trainSamples1 = trainSamples[:, 0:dim]
trainConceptnet1 = trainSamples1[:, 0:1]

trainSamples2 = trainSamples[:, dim:(dim*2)]
trainConceptnet2 = trainSamples2[:, 0:1]

testSamples1 = testSamples[:, 0:dim]
testConceptnet1 = testSamples1[:, 0:1]

testSamples2 = testSamples[:, dim:(dim*2)]
testConceptnet2 = testSamples2[:, 0:1]

conceptnet1 = tf.placeholder(tf.float32, [None, 1])
conceptnet2 = tf.placeholder(tf.float32, [None, 1])

A = tf.constant(1.0)
B = tf.constant(0.0)
C = tf.constant(0.0)

value1 = conceptnet1 * (A + conceptnet1 * (B + C * conceptnet1))
value2 = conceptnet2 * (A + conceptnet2 * (B + C * conceptnet2))

simDiff = value1 - value2

invstddev = tf.Variable(0.0)
S1 = tf.Variable(1.0)
S2 = tf.constant(0.0)
stddev1 = S1 + conceptnet1 * S2
stddev2 = S1 + conceptnet2 * S2
stddev = tf.sqrt(stddev1*stddev1 + stddev2*stddev2)

y = (tf.erf(simDiff / (math.sqrt(2.0) * stddev))+1.0)/2.0

cross_entropy = tf.reduce_mean(-tf.log(y))

train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
  sess.run(train_step, feed_dict={conceptnet1: trainConceptnet1, conceptnet2: trainConceptnet2})
  res = sess.run([cross_entropy, S1, S2, B, C], feed_dict={conceptnet1: testConceptnet1, conceptnet2: testConceptnet2})
  lastAns = res[0]
  stddev = res[1]

print("Standard deviation: " + str(stddev))

if len(sys.argv) >= 4:
  f = open(sys.argv[3], 'w')
  f.write(str(lastAns))
  f.close()
  sys.exit()
