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


  


  


















