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




















