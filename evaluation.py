import numpy as np
from numpy import array
import matplotlib.pyplot as plt

import os
from PIL import Image
from keras.preprocessing import image
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array
import pickle
import json
import requests
from io import BytesIO
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import numpy

########################################################################
### methods go here ###
def loadImage(URL):
    try:
        response = requests.get(URL)
        # convert the image pixels to a numpy array
        img = Image.open(BytesIO(response.content)).resize((224,224))
        return img_to_array(img)
    except:
        return None

def openImage(URL):
    try:
        response = requests.get(URL)
        # convert the image pixels to a numpy array
        img = Image.open(BytesIO(response.content)).resize((224,224))
        ####v2
        # img = Image.open(BytesIO(response.content)).resize((299,299))
        return img
    except:
      return None

def max_length(descriptions):
#     lines = to_lines(descriptions)
    for d in descriptions:
        print(len(d.split()))
    return max(len(d.split()) for d in descriptions)


def create_sequences(poems, photos, word2index, max_poem_len, num_photos_per_batch):
    X1, X2, y = list(), list(), list()

    for i in range(len(photos)):
        # retrieve the photo feature
        photo = photos[i]
        poem = poems[i]

        # encode the sequence
        seq = [word2index[word] for word in poem.split(' ') if word in word2index]
        print(seq)

        # split one sequence into multiple X, y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_poem_len)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

            # store
            X1.append(photo[0])
            X2.append(in_seq)
            y.append(out_seq)

    return array(X1), array(X2), array(y)


def pickle_this(data, fname):
    try:
      with open(fname, 'wb') as tosave:
        pickle.dump(data, tosave)
    except:
      print('wasnt able to pickle')

########################################################################
### starting program here ###

with open("test_poems.json") as json_data:
    data = json.load(json_data)

#############################
# load the model
model = VGG16()

# re-structure the model
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

####v2
# model = InceptionV3(weights='imagenet')
# model_new = Model(model.input, model.layers[-2].output)

#print(model.summary())

#############################

vocab = set()
poems = list()
seq_poems = list()
img_names = list()
features= dict()
featurelist = list()
#############################

# extract features from each photo
for val in data:
    print(val['image_url'])
    # load an image from file
    image = loadImage(val['image_url'])
    if image is not None and image.shape == (224,224,3):

        img_names.append(val['image_url'])
        poems.append(val['poem'])
        seq_poems.append('***startseq '+val['poem']+' endseq***')
        vocab.update(val['poem'].split())

#             print(image.shape)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
#             print(image.shape)
        image = preprocess_input(image)

        # get features
        feature = model.predict(image, verbose=0)
        featurelist.append(feature)
        # get image id
        image_id = val['id']

        # store feature
#        features[image_id] = feature

        print('>%s' % image_id)
    else:
#        print('pass')
        pass

# with open('featurelist.pickle', 'wb') as tosave:
#     pickle.dump(featurelist, tosave)


print('Number of poems: ', len(img_names))
print('Number of poems: ', len(poems))
print('Original Vocabulary Size: %d' % len(vocab))

#############################
with open(r"vocab.pickle", "rb") as input_file:
  vocab = pickle.load(input_file)
index2word = {}
word2index = {}

i = 1
for word in vocab:
    word2index[word] = i
    index2word[i] = word
    i += 1

max_poem_len = max(len(p) for p in seq_poems)
vocab_size = len(index2word) + 1

#############################

glove_dir = 'glove.6B.200d.txt'
embeddings_index = {} # empty dictionary
f = open(os.path.join('glove.6B.200d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_dim = 200
# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))
# print(embedding_matrix[441])

for word, i in word2index.items():
    embedding_vector = embeddings_index.get(word)
#    print(embedding_vector)
    if embedding_vector is not None:
#         print(i)
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector




def generate_desc(photo, word2index, index2word, model, max_length):

    in_text = 'startseq***'
    for i in range(max_length):
        sequence = [word2index[w] for w in in_text.split() if w in word2index]
        sequence = pad_sequences([sequence], maxlen=max_length)


        yhat = model.predict([photo[0],sequence], verbose=0)
        # yhat = np.argmax(yhat)
        # print(yhat)
        diversity = 0.2
        
        index = select(yhat, diversity)

        word = index2word[index]
        in_text += ' ' + word
        if word == 'endseq***':
          break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final




def evaluate_model(model, poems, featurelist, max_length, word2index, index2word, img_names):
    actual, predicted = list(), list()
    # step over the whole set
    for i in range(3):
      photo = featurelist[i]
      poem = poems[i]
      img = openImage(img_names[i])
      plt.imshow(img)
      plt.show()

      # generate description
      yhat = generate_desc(featurelist, word2index, index2word, model, 720)
      print(yhat)
      # store actual and predicted
      actual.append(poem)
      predicted.append(yhat)
        
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))



import numpy as np
def select(preds, temperature=1.0):
# helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds[0], 1)
    return np.argmax(probas)

import matplotlib.pyplot as plt
import pickle

with open(r"featurelist.pickle", "rb") as input_file:
  featurelist1 = pickle.load(input_file)
  
model1 = load_model('model-ep002-loss6.484-val_loss6.704.h5')

evaluate_model(model1, poems, featurelist1, max_poem_len, word2index, index2word, img_names)
# evaluate_model(model2, poems, featurelist2, max_poem_len, word2index, index2word)

