# Machine-Translation
Behind the language translation services are the machine translation models.

### Introduction to Machine translation
- Dataset used are English and French sentences text files, where each line in English file consists an english sentence and the other french file consists the corresponding french translation of english sentences.
- In Machine translation terminology, the language to be translated is called the source language and the translated language is called the target language.
- **Machine translation** : First the words of the source sentence are fit to the model one-by-one, sequentially. Then the model outputs the predicted translation word-by-word in a sequential manner.

#### One-hot encoded vectors
- When feeding words to a machine translation model, words needs to be converted to a numerical representation.One hot encoding is one of the commonly used transformations.
- In one hot encoding, a word is represented as a vector of zeros and ones. The length of the vector is determined by the size of the vocabulary. The vocabulary is the collection of unique words used in the dataset for a specific language.
- In `keras` we can use **`to_categorical`** function to convert words to onehot encoded vectors.However, inorder to use this function, we first need to convert individual words to integers or IDs.To do that we define a Python dictionary that maps words to integers.

```python
# a mapping containing words and their corresponding indices
word2index = {"i":0, "like":1, "cats":2}

# convert words to IDs or indices
words=["i", "like", "cats"]
word_ids = [word2index[w] for w in words]
print(word_ids)

# by passing these word_ids to `to_categorical` function we can obtain the one-hot vectors. If we dont pass the length of vector, keras will automatically detect it from the data we pass

onehot_1 = to_categorical(word_ids)
print([(w,ohe.tolist()) for w, ohe in zip(words, onehot_1)])

# but we can fix the length of the vector by passing the `num_classes` argument
onehot_2 = to_categorical(word_ids, num_classes=5)
print([(w, ohe.tolist()) for w, ohe in zip(words, onehot_2)])
```
### Encoder Decoder Architecture
- A machine translation model works by, first consuming the words of the source language sequentially, and then sequentially predicting the corresponding words in the target language.However, under the hood, it is actually two different models; an encoder and a decoder.
- In our case, encoder takes one hot representation of English words as inputs and produces a compressed representation of the words known as a context vector. Then, the decoder consumes the context vector as an input and produces probabilistic predictions for each time step.
- The word for a given time step is selected as the word with the highest probability.
- Although, the inputs to the encoder are ones and zeros, the decoder produces continuos probabilistic outputs. These models are also called sequence to sequence models because they map a sequence that is, an English sentence to another sequence, that is a French sentence.

#### Reversing sentences - encoder decoder model
- A simple model that reverses a sentence. First, the encoder receives an one-hot representation of the sentence and converts it to word IDs.
- Next, the decoder takes in the word IDs, reverses them and converts the reversed IDs back to the one-hot representation resulting in the reversed sentence.

#### Writing the encoder

```python
def words2onehot(word_list, word2index):
  """converts given list of words to one hot vectors"""
  word_ids = [word2index[w] for w in word_list]
  onehot = to_categorical(word_ids, 3)
  return one_hot
```
- The `encoder` function is a simple function, that takes in an array of one hot vectors as the argument and returns the word IDs corresponding to the one-hot vectors.

```python
def encoder(onehot):
  word_ids = np.argmax(onehot axis=1)
  return word_ids
  
onehot = words2onehot(["I", "like", "cats"], word2index)
context = encoder(onehot)
print(context)
```
- The context contains the corresponding word IDs of the words.

#### Writing the decoder
- The decoder takes the word IDs, reverse the IDs and then returns the one-hot vectors of the reversed words

```python
def decoder(context_vector):
  word_ids_rev = context_vector[::-1]
  onehot_rev = to_categorical(word_ids_rev, 3)
  return onehot_rev
```

- Decoder takes one-hot vectors as input produced by the encoder and produces one-hot vectors of the reversed words.
- Helper function : onehot2words - which converts a set of one-hot vectors to human readable words.

```python
def onehot2words(onehot, index2word):
  ids = np.argmax(onehot, axis=1)
  return [index2word[id] for id in ids]
 
onehot_rev = decoder(context)
reversed_words = onehot2words(onehot_rev, index2word)
print(reversed_words)
```

### Sequential models
#### Time series inputs and sequential models
- A sentence is an time series input which means every word in the sentence is affected by previous words. The encoder and decoder use a ML model that can learn from time-series or sequential inputs like sentences. The ML model is called a sequential model.

#### Sequential models
- Sequential models go from one input to the other while producing an output at each time step.During time step 1,the first word is processed and during time step 2, the second word is processed.The same model processes each input.

#### Encoder as a sequential model
- Type of sequential model called gated recurrent unit(GRU), is used as translator. For e.g, the inputs to the encoder is a sequence of English words encoded as one-hot vectors.

#### Keras(Functional API)
- Keras has two important objects: Layers and Models.
- Input Layer : `inp = keras.layers.Input(shape=(...))`
- Hidden layer: `layer = keras.layers.GRU(...)`
- Output      : `out = layer(inp)`
- Model       : `model = Model(inputs=inp, outputs=out)`

#### Understanding the shape of the data
- Before getting to implementing GRUs, we must understand that sequential data has three dimensions. The sequences are usually processed in groups or batches.
- Sequential data is 3-dimensional
1) Batch dimension (e.g batch=groups of sentences)
2) Time dimension(sequence length) : describes the length of the sequences or sentences
3) Input dimension: length of one-hot vectors
- The input layer of the GRU model needs to have this 3 dimensional shape.

#### Implementing GRUs with keras

```python
inp = keras.layers.Input(batch_shape=(2,3,4)) #batchsize=2, sequence length=3, input dimen=4
gru_out = keras.layers.GRU(10)(inp)  #GRU layer with 10 hidden units
model = keras.models.Model(inputs=inp, outputs=gru_out)

x = np.random.normal(size=(2,3,4))
y = model.predict(x)
print("shape (y) = ", y.shape, "\ny = \n", y)
```

- we can also define the input layer by setting the batch size to None, to do that use the shape argument instead of batch_shape and only set the sequence length and input dimensionality
- In keras, doing this means that the input layer will accept any arbitary sized batch of data.This allows to define the keras model once and experiment with different bacth sizes without changing the model.
- GRU layer has two more important arguments return_state and return sequences.If we set the `return_state` argument to True, the model will return two outputs instead of one, one is the last hidden state and the other is the last output.
- If we set `return_sequences` to True the model will output all the outputs in the sequence of the last output.

### Implementing the encoder
#### Understanding the data
- Dataset consists of english and french sentences list

```python
for en_sent, fr_sent in zip(en_text[:3], fr_text[:3]):
  print('english:', english)
  print('french:', french)
```
- Check attributes of the dataset such as, the average number of words, or average length of sentences, and the size of the vocabulary.These parameters are required to define the input layer of the encoder.
- Tokenazation can be done by splitting the sentences by space.

#### Computing the length of the sentences

```python
sent_lengths = [len(en_sent.split(" ")) for en_sent in en_text]
mean_length = np.mean(sent_lengths)
```
#### Computing the size of the vocabulary

```python
all_words = []
for sent in en_text:
  all_words.extend(sent.split(" "))
vocab_size = len(set(all_words))
```

#### The Encoder
- Encoder is made up of GRU model.The GRU model goes from one input to the other sequentially while producing an output(and a state) at each time step.
- The state vector produced at time equals t becomes an input state to the model at time equals t plus 1.
- Encoder implementation is very similar to implementation of GRU layer
- Knowing the **average number of words** helps us to define **`en_len`**. The **size of the vocabulary** helps us to define **`en_vocab`**. These are essential to define the input layer

```python
# Input layer
en_inputs = Input(shape=(en_len, en_vocab))
```

- We picks these values close to what we discovered while analyzing the dataset
- Next, define a GRU layer which returns the last state.The last state of the GRU layer will be later passed to the decoder as inputs.

```python
# GRU layer
en_gru = GRU(hsize, return_state=True)
en_out, en_state = en_gru(en_inputs)
```
- Next define a keras model whose input is input layer and output is the state obtained from the GRU layer

```python
# Keras model
encoder = Model(inputs=en_inputs, outputs=en_state)
```
### Decoder
- We have implemented a encoder which consumes the source i.e english words one by one and finally produces the context vector.Now we need to implement the decoder that consumes the context vector and produces the target language words i.e french words one by one. The decoder will be implemented similar to the encoder using a Keras GRU layer. But GRU layer requires two inputs 1) A time series input & 2) Hidden state.



















































  


















