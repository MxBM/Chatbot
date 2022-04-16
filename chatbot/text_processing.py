from asyncore import read
import os
import json
import unicodedata
import re

"""

These flags are utilized to replace the input textual data with these flags that the encoder and decoder need to create an embedding representation
of the sequence. 

Refer here: https://datascience.stackexchange.com/questions/26947/why-do-we-need-to-add-start-s-end-s-symbols-when-using-recurrent-neural-n

The padding is implemented to ensure that all layers performing matrix multiplication have the same dimensional vectors, we can't have vectors being multiplied
with different numbers of rows and columns, it creates irregularites.
Refer here: https://stackoverflow.com/questions/57393033/why-do-we-need-padding-in-seq2seq-network

"""
PAD_FLAG = 0  # Padding for shorter sentences
SOS_FLAG = 1  # Start of sentence
EOS_FLAG = 2  # End of sentence


class Vocab:  # Preprocess data to be used in a seq2seq model
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_FLAG: "PAD", SOS_FLAG: "SOS", EOS_FLAG: "SOS"}
        self.num_words = 3

        # Iterate through given sentence and split on spaces to obtain a word.
        def addSentence(self, sentence):
            for word in sentence.split(' '):
                self.addWord(word)  # Call addWord function

        # Add words to the vocab object
        def addWord(self, word):
            if word not in self.word2index:  # If the word isn't in the vocab object yet
                self.word2index[word] = self.num_words
                self.word2count[word] = 1
                self.index2word[self.num_words] = word
                self.num_words += 1
            else:  # Otherwise it's already in the vocab object
                self.word2count[word] += 1

        # Exclude words from our Vocab depending on how often they show up in our dataset
        def trim(self, min_count):
            # If the word has already been trimmed (flag is true)
            if self.trimmed:
                return
            self.trimmed = True  # Set to true since it's trimmed.

            selected_words = []

            # Iterate through word2count array and find words that appear below a certain threshhold (min_count) provided when the function is called
            for x, y in self.word2count.items():  # Tuple unpacking. Refer here: https://stackoverflow.com/questions/59117667/for-x-y-in-function-explain-the-python-for-loop-structure
                if y >= min_count:  # Word appears more than threshold
                    # Meaning the word can be added to the vocabulary.
                    selected_words.append(x)

            print('selected_words {} / {} = {:4f}'.format(
                len(selected_words), len(self.word2index), len(
                    selected_words) / len(self.word2index)
            ))

            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_FLAG: "PAD",
                               SOS_FLAG: "SOS", EOS_FLAG: "SOS"}
            self.num_words = 3

            for word in selected_words:  # From list of accepted words, call the addWord method to add word to vocabulary
                self.addWord(word)


datafile = os.path.join(".", "full_data.txt")  # Path where data
MAX_LEN = 12  # Max sentence length to parse out

# Open JSON file of contractions in read mode.
with open('contractions.json', 'r') as fp:
    fixed_prefix = json.load(fp)  # Get JSON file and return it as an object
    """
    The JSON object consists of contractions with the key being the contraction and the value being the simplified word, 
    (EX: "ain't" is the key and "am not" is the value). Since all of they keys are lowercase we have to strip all words 
    """
    fixed_prefix = {key.strip().lower(): value.strip().lower()
                    for key, value in fixed_prefix}


# To make things a hell of a lot easier we don't want our training set to have any strings composed of unicode characters, we want plain ASCII, no accents, or other characters.
# Refer here: https://docs.python.org/3/library/unicodedata.html
# Refer here: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string/518232#518232
def unicode2Ascii(string):
    return ''.join(
        # Normalize the given Unicode string with Normal Form D which is known as canonical decomposition which translates each character into it's decomposed form"
        c for c in unicodedata.normalize('NFD', string)
        if unicodedata.category(c) != 'Mn'  # Refer Here
    )


# This function enforces lowercase letters, removing spaces with trim and removing any strange symbols such as ! , ?, etc. This is doen with Regex.
def normalizeString(string):
    # Pass in the string to be decomposed into ascii code, ensuring that we pre-process it into lowercase and removing whitespace
    string = unicode2Ascii(string.lower().strip())
    string = " ".join(
        [fixed_prefix[each] if each in fixed_prefix else each for each in string.split()])
    # Use regex to match ! . and ? symbols to in the given string and remove it. Refer here: https://docs.python.org/3/library/re.html
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
    # Obtain a pattern with the given regex to be found in our string
    pattern = re.compile(r"(.)\1{2,}")
    # Utilize the pattern to find instances of lowercase letters, and symbols that shouldn't be in the string
    string = pattern.sub(r"\1\1", string)
    string = re.sub(r"\s+", r" ", string).strip()
    return string  # Clean String


def returnVocab(datafile, dataset):
    print('PARSING DATA..')

    # The file is read into lines.
    lines = open(datafile, encoding='utf-8').\
        read().strip().split("\n")

    # Not really sure wtf this line is doing. I know it's normalizing the strings in each line and splitting on <CoSe> but I'm not sure where that comes fromm, maybe denoting the end of a line?
    pairs = [[normalizeString(s) for s in l.split('<CoSe>')] for l in lines]

    vocabulary = Vocab(dataset)  # Create a Vocab object

    return vocabulary, pairs


# Function to check if sentences in a pair adhere to the MAX_LEN chracter limit. This is done to preserve memory usage and speedup training.
def filterPair(pair):
    # Input sequences need to preserve the last word for EOS token
    if len(pair) == 2:
        # Check if each pair (question & response) adhere to the rule
        return len(pair[0].split(' ')) < MAX_LEN and len(pair[1].split('  ')) < MAX_LEN


# Function to filter pairs using filterPair
def filterPairs(pairs):
    # Essentially parse out the pairs that don't surpass the character limit
    return [pair for pair in pairs if filterPair(pair)]


# Prepare the vocabulary by loading everything required
def loadPrepareData(dataset, datafile, directory):
    print("PREPARING TRAIN DATA...")
    vocabulary, pairs = returnVocab(datafile, dataset)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        vocabulary.addSentence(pair[0])
        vocabulary.addSentence(pair[1])
    print("Counted words:", vocabulary.num_words)
    return vocabulary, pairs


save_dir = os.path.join(".", "model", "save")
corpus_name = 'chatbot'
voc, pairs = loadPrepareData(corpus_name, datafile, save_dir)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)
