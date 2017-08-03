from keras.layers import Input, Dense, Lambda, Add, Activation, Multiply, Concatenate, Dropout
from keras.models import Model
from keras import regularizers
import numpy as np
import sys

if len(sys.argv) < 3 or len(sys.argv) > 4:
  print("Usage: python3 training.py train-file test-file [result-file]")
  sys.exit()

# Parse samples
trainSamples = np.loadtxt(sys.argv[1], ndmin=2)
# Dimension of features of one word (there are two words in each sample)
dim = trainSamples.shape[1]//2
# Dimension of meta-features of each word
metaDim = 4
# Dimension of word2vec-vector of each word
wordVecDim = (dim-metaDim)//2
trainOutput = np.ones(trainSamples.shape[0])
testSamples = np.loadtxt(sys.argv[2], ndmin=2)
testOutput = np.ones(testSamples.shape[0])
trainSamples1 = trainSamples[:, 0:dim]
trainSamplesMeta1 = trainSamples1[:, 0:metaDim]
trainSamplesVec1 = trainSamples1[:, metaDim:(metaDim+wordVecDim)]
trainConceptnet1 = trainSamples1[:, 0:1]
trainSamples2 = trainSamples[:, dim:(dim*2)]
trainSamplesMeta2 = trainSamples2[:, 0:metaDim]
trainSamplesVec2 = trainSamples2[:, metaDim:(metaDim+wordVecDim)]
trainConceptnet2 = trainSamples2[:, 0:1]
trainSamplesClue = trainSamples[:, (metaDim+wordVecDim):dim]
testSamples1 = testSamples[:, 0:dim]
testSamplesMeta1 = testSamples1[:, 0:metaDim]
testSamplesVec1 = testSamples1[:, metaDim:(metaDim+wordVecDim)]
testConceptnet1 = testSamples1[:, 0:1]
testSamples2 = testSamples[:, dim:(dim*2)]
testSamplesMeta2 = testSamples2[:, 0:metaDim]
testSamplesVec2 = testSamples2[:, metaDim:(metaDim+wordVecDim)]
testConceptnet2 = testSamples2[:, 0:1]
testSamplesClue = testSamples[:, (metaDim+wordVecDim):dim]

# Set up model
wordMeta1 = Input((metaDim,))
wordMeta2 = Input((metaDim,))
wordVec1 = Input((wordVecDim,))
wordVec2 = Input((wordVecDim,))
clueVec = Input((wordVecDim,))
conceptnet1 = Input((1,))
conceptnet2 = Input((1,))

sim1 = Multiply()([wordVec1, clueVec])
sim2 = Multiply()([wordVec2, clueVec])

dim_s = 1
S = Dense(dim_s, input_shape=[wordVecDim], activation='elu',
    kernel_regularizer=regularizers.l2(0.03))
DS = Dropout(0.0)
s1 = DS(S(sim1))
s2 = DS(S(sim2))

t1 = Concatenate()([wordMeta1, s1])
t2 = Concatenate()([wordMeta2, s2])

dim_a = 20
A = Dense(dim_a, input_shape=[metaDim+dim_s], activation='elu',
    kernel_regularizer=regularizers.l2(0.0004))
DA = Dropout(0.0)
a1 = DA(A(t1))
a2 = DA(A(t2))

dim_b = 20
B = Dense(dim_b, input_shape=[dim_a], activation='elu')
b1 = B(a1)
b2 = B(a2)

dim_c = 1
C = Dense(dim_c, input_shape=[dim_b], activation='elu')
c1 = C(b1)
c2 = C(b2)

score1 = c1
score2 = c2

# This provides a simple reference value, based solely on scalar products from conceptnet
Reference = Dense(1, input_shape=[1])
#score1 = Reference(conceptnet1)
#score2 = Reference(conceptnet2)

minus_score2 = Lambda(lambda x: -x)(score2)
score_diff = Add()([score1,minus_score2])
probability = Activation('sigmoid')(score_diff)

model = Model(inputs=[wordMeta1, wordMeta2, wordVec1, wordVec2, clueVec, conceptnet1, conceptnet2], outputs=probability)

model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# Train model
model.fit([trainSamplesMeta1, trainSamplesMeta2, trainSamplesVec1, trainSamplesVec2, trainSamplesClue, trainConceptnet1, trainConceptnet2], trainOutput, epochs=1000, batch_size=500)

# Evaluate model
score = model.evaluate([testSamplesMeta1, testSamplesMeta2, testSamplesVec1, testSamplesVec2, testSamplesClue, testConceptnet1, testConceptnet2], testOutput)
print("\nCross entropy: " + str(score[0]))
print("Accuracy: " + str(score[1]))

if len(sys.argv) >= 4:
  f = open(sys.argv[3], 'w')
  f.write(str(score[0]))
  f.close()
  sys.exit()

# Read samples from stdin to test the model
while(True):
  clue = input()
  word1 = input()
  word2 = input()

  from subprocess import Popen, PIPE

  p = Popen(['./codenames', '--extract-features', 'predict.txt'], stdout=PIPE, stdin=PIPE)

  p.stdin.write(bytes(clue + " : " + word1 + " " + word2, 'UTF-8'))
  p.communicate()
  p.stdin.close()

  testSamples = np.loadtxt("predict.txt", ndmin=2)
  if(testSamples.shape[0] == 0):
    print("There was an unknown word")
    continue
  testOutput = np.ones(testSamples.shape[0])
  testSamples1 = testSamples[:, 0:dim]
  testSamplesMeta1 = testSamples1[:, 0:metaDim]
  testSamplesVec1 = testSamples1[:, metaDim:(metaDim+wordVecDim)]
  testConceptnet1 = testSamples1[:, 0:1]
  testSamples2 = testSamples[:, dim:(dim*2)]
  testSamplesMeta2 = testSamples2[:, 0:metaDim]
  testSamplesVec2 = testSamples2[:, metaDim:(metaDim+wordVecDim)]
  testConceptnet2 = testSamples2[:, 0:1]
  testSamplesClue = testSamples[:, (metaDim+wordVecDim):dim]
  result = model.predict([testSamplesMeta1, testSamplesMeta2, testSamplesVec1, testSamplesVec2, testSamplesClue, testConceptnet1, testConceptnet2])[0][0]
  print("%.2f %% that a human would pick %s" % ((result*100), word1))
