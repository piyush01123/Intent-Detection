# Intent Detection Keras Implementation
Intent Detection on ATIS dataset

## Required Libraries:
```
tensorflow
keras
collections
json
numpy
scikit-learn
```

## Reproduction of work
Download Train and test datasets
```bash
wget https://github.com/piyush-kgp/Intent-Detection/releases/download/1.0/atis-2.train.w-intent.iob.3.txt
wget https://github.com/piyush-kgp/Intent-Detection/releases/download/1.0/atis.test.w-intent.iob.2.txt
```

---
Download Word Embedding file from internet
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

---
Run Preprocessing, Get Frequency Summary, Train the model and view result
```bash
python3 prepro.py
python3 count_frequency.py
python3 train.py
```
---
## Code Walkthrough
`prepro.py` does the preprocessing and creates train and test JSONs.
This step converts the data into a JSON format which is essentially
a list of Python dict objects. Each dict is in the form:
```
{'sentence': _,
  'intent': _,
  'subintents': _
}
```

`subintents` is again a dict of several subintents.


`count_frequency.py` gives the class probabilities of all intents


`tokenize_.py` takes the word embedding text file and returns a proper Python
 dictionary object with keys of word and values of 300-dimensional embeddings in the form
 of Numpy array of dimension (300,)

 `train.py`  trains the model and shows accuracy.
