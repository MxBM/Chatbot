import os
import json
import pickle
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


datafile = os.path.join(
    ".", "chatbot/formatted_movie_lines.txt")  # Path where data
MAX_LEN = 12  # Max sentence length to parse out

# Open JSON file of contractions in read mode.
with open('chatbot\contractions.json', 'r') as fp:
    fixed_prefix = json.load(fp)  # Get JSON file and return it as an object

    fixed_prefix = {key.strip().lower(): value.strip().lower()
                    for key, value in fixed_prefix.items()}


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
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    vocabulary = Vocab(dataset)  # Create a Vocab object

    return vocabulary, pairs


# Function to check if sentences in a pair adhere to the MAX_LEN chracter limit. This is done to preserve memory usage and speedup training.
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    if len(p) == 2:
        # Check if each pair (question & response) adhere to the rule
        return len(p[0].split(' ')) < MAX_LEN and len(p[1].split('  ')) < MAX_LEN


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
    for pair in pairs:  # Iterate through pairs
        vocabulary.addSentence(pair[0])  # Zero index is input sentence
        vocabulary.addSentence(pair[1])  # First index is the output sentence
    print("Counted words:", vocabulary.num_words)  # Total number of words
    return vocabulary, pairs


MIN_COUNT = 3  # Minimum word count for trimming.


def removeRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT form the vocab object voc
    voc.trim(MIN_COUNT)  # Refer to Vocab Trim function
    keep_pairs = []  # Pairs to keep
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Iterate through input sentences
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Iterate through output sentences
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only append pairs to new pair list if they both don't contain any trimmed words in thier input sent. or output sent.
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(
        len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))

    return keep_pairs


# TODO: IMPLEMENT FUNCTION TO REMOVE NAMES FROM DATASETS AS THEY'RE NOT USEFUL

save_dir = os.path.join(".", "model", "save")
corpus_name = 'chatbot'
voc, pairs = loadPrepareData(corpus_name, datafile, save_dir)
pairs = removeRareWords(voc, pairs, MIN_COUNT)


# Pickle allows for the serilization of a python object structure so that it can be saved on disk and later read a byte stream/data stream.
# This is required to save the vocab object which is used later on in the encoder and decoder model.
# Refer here for Pickle Information: https://www.journaldev.com/15638/python-pickle-example#:~:text=Python%20Pickle%20dump&text=dump()%20function%20to%20store,that%20you%20want%20to%20store.

# Open the path ./pickles/vocab.file in binary mode as f
with open(os.path.join(".", "chatbot\pickles", "vocab.file"), "wb") as f:
    pickle.dump(voc, f, pickle.HIGHEST_PROTOCOL)
    # First argument is the object to be stored
    # Second argument is the file object you get by opening the desired file in wb mode (write-binary mode)
    # Third argument is simply the protocal utilized for specific python versions loading data. Refer here: https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice

# Read Binary Mode to load dumped pickle file above.
with open(os.path.join(".", "chatbot\pickles", "vocab.file"), "rb") as f:
    dump = pickle.load(f)
print(dir(dump))
