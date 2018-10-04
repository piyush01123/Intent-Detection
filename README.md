# Intent-Detection
Intent Detection on ATIS dataset

## Required Libraries:
```
keras
collections
json
numpy
scikit-learn
```

# Reproduction of work
Download Word Embedding file from internet
`wget http://nlp.stanford.edu/data/glove.6B.zip`

Unzip files
`unzip glove.6B.zip`

Train the model and view result
`python3 train.py`


# Code Descriptions
Pre-processing: This converts the data into a JSON format which is essentially
a list of Python dict objects. Each dict is in the form:
`{'sentence': _,
  'intent': _,
  'subintents': _
}`
`subintents` is again a dict of several subintents.

Run on terminal
`python3 prepro.py`. This does the preprocessing and creates train and test JSONs.

`python3 count_frequency.py` gives the class probabilities of all intents

`python3 tokenize_.py` takes the word embedding text file and returns a proper Python
 dictionary object with keys of word and values of 300-dimensional embeddings in the form
 of Numpy array of dimension (300,)

 Train the model and view result
 `python3 train.py`
