# Class For Encoder Recurrent Neural Network
# Utilize base class for neural network modules (torch) refer here: https://pytorch.org/docs/stable/generated/torch.nn.Module.html


# Understand GRU's: https://www.analyticsvidhya.com/blog/2021/03/introduction-to-gated-recurrent-unit-gru/

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import itertools
import pickle
import unicodedata
import json
import re
import matplotlib.pyplot as plt
from torch import optim

CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

PAD_FLAG = 0  # Padding for shorter sentences
SOS_FLAG = 1  # Start of sentence
EOS_FLAG = 2  # End of sentence
MAX_LEN = 10
MIN_COUNT = 3  # Minimum word count for trimming.


class Vocab:  # Preprocess data to be used in a seq2seq model

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_FLAG: "PAD", SOS_FLAG: "SOS", EOS_FLAG: "EOS"}
        self.num_words = 3

    # Iterate through given sentence and split on spaces to obtain a word and add it to the vocab.
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

        selected_words = []  # Array of words to keep

        # Iterate through word2count array and find words that appear below a certain threshhold (min_count) provided when the function is called
        for x, y in self.word2count.items():  # Tuple unpacking. Refer here: https://stackoverflow.com/questions/59117667/for-x-y-in-function-explain-the-python-for-loop-structure
            if y >= min_count:  # Word appears more than threshold
                # Meaning the word can be added to the vocabulary.
                selected_words.append(x)

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_FLAG: "PAD",
                           SOS_FLAG: "SOS", EOS_FLAG: "EOS"}
        self.num_words = 3

        for word in selected_words:  # From list of accepted words, call the addWord method to add word to vocabulary
            self.addWord(word)


datafile = os.path.join(
    ".", "chatbot/formatted_movie_lines.txt")  # Path where data

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
    # pattern = re.compile(r"(.)\1{2,}")
    # Utilize the pattern to find instances of lowercase letters, and symbols that shouldn't be in the string
    # string = pattern.sub(r"\1\1", string)
    string = re.sub(r"\s+", r" ", string).strip()
    return string  # Clean String


def returnVocab(datafile, dataset):
    print('...creating vocab object')
    # The file is read into lines.
    lines = open(datafile, encoding='utf-8').\
        read().strip().split("\n")
    # Normalize the strings in the pair
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # Create a Vocab object from the constructed dataset
    vocabulary = Vocab(dataset)
    return vocabulary, pairs


# Function to check if sentences in a pair adhere to the MAX_LEN chracter limit. This is done to preserve memory usage and speedup training.
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LEN and len(p[1].split(' ')) < MAX_LEN


# Function to filter pairs using filterPair
def filterPairs(pairs):
    # Essentially parse out the pairs that don't surpass the character limit
    return [pair for pair in pairs if filterPair(pair)]


# Prepare the vocabulary by loading everything required
def loadPrepareData(corpus_name, datafile, directory):
    print("...preprocessing")
    vocabulary, pairs = returnVocab(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("...counting words")
    for pair in pairs:  # Iterate through pairs
        vocabulary.addSentence(pair[0])  # Zero index is input sentence
        vocabulary.addSentence(pair[1])  # First index is the output sentence
    print("Counted words:", vocabulary.num_words)  # Total number of words
    return vocabulary, pairs


# TODO: IMPLEMENT FUNCTION TO REMOVE NAMES FROM DATASETS AS THEY'RE NOT USEFUL

# Directory where new txt file is going to be saved
save_dir = os.path.join(".", "model", "save")
corpus_name = 'chatbot'
voc, pairs = loadPrepareData(
    corpus_name, datafile, save_dir)  # Obtain preped data


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(
        len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


pairs = trimRareWords(voc, pairs, MIN_COUNT)


# Pickle allows for the serilization of a python object structure so that it can be saved on disk and later read a byte stream/data stream.
# This is required to save the vocab object which is used later on in the encoder and decoder model.
# Refer here for Pickle Information: https://www.journaldev.com/15638/python-pickle-example#:~:text=Python%20Pickle%20dump&text=dump()%20function%20to%20store,that%20you%20want%20to%20store.
# Open the path ./pickles/vocab.file in binary mode as f
with open(os.path.join(".", "chatbot\pickles", "vocab.file"), "wb") as f:
    pickle.dump(voc, f, pickle.HIGHEST_PROTOCOL)
    # First argument is the object to be stored
    # Second argument is the file object you get by opening the desired file in wb mode (write-binary mode)
    # Third argument is simply the protocal utilized for specific python versions loading data. Refer here: https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
print('...serializing vocab object')

# Read Binary Mode to load dumped pickle file above.
with open(os.path.join(".", "chatbot\pickles", "vocab.file"), "rb") as f:
    dump = pickle.load(f)
print('...loading serialized object')
print(dir(dump))


# Obtain indices from provided sentence in the voc object, ensuring that the EOS_FLAG index is denoted
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_FLAG]

# All sentences need to be padded to a constant size, (index should maintain the PAD_FLAG which denotes padding)


def zeroPadding(l, fillvalue=PAD_FLAG):
    # Easy way to aggregate elements from each sentence iterated if of uneven length.
    # Refer here: https://docs.python.org/3/library/itertools.html#itertools.zip_longest
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


# Constructs a matrix of input tensor/lengths, ensures that it's padded
def binaryMatrix(l, value=PAD_FLAG):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_FLAG:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Convert passed in sentence to be converted to a tensor (matrix) that's zero padded
def inputVar(sentences, voc):
    # Obtain batch of sentences to be converted to tensors
    indexes_batch = [indexesFromSentence(
        voc, sentence) for sentence in sentences]
    # Use torch to change entire batch of sentences to a tensor shape
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Make sure to pad them since not every sentence will be of equal length
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths  # Return a tensor of lenghs for each batch.


# Return a tensor mask that has the same shape as the output for the inputVar function,
# goal is to return a mask that overlays every element that has a PAD_FLAG as a zero and all of the others as one.
# Refer here: https://jmlb.github.io/ml/2017/12/06/create_mask_tf/
def outputVar(sentences, voc):
    indexes_batch = [indexesFromSentence(
        voc, sentence) for sentence in sentences]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Function to take in pairs of sentences converts them to matricies that are padded.
def batch2TrainData(voc, pair_batch):
    # Sort the batches in descending order.
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []  # Array of batches
    # Iterate through batches and append each array into their own respective input/output batches. One for the mask (output) and one for the tensor shape (input)
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


class EncoderRNN(nn.Module):
    # Allows us to reference EncoderRNN obj with the use of indirection. Refer here: https://www.programiz.com/python-programming/methods/built-in/super
    # The Encoder Mechanism is defined in a separate class
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers  # Denote number of layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        # Load up Pytorch GRU module from NN. Gated Recurrent unit is an alternate version of an LSTM (Long Short Term Memory RNN)
        # Think of them as simplified logic gates to save and throw away input data. We set the birectional flag to true.
        # Thus a birectional GRU is being used which utilizes the input sequence which is reverse and passed onto another GRU
        # Refer here: https://blog.paperspace.com/bidirectional-rnn-keras/

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        # because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # We first must convert the text input and embed it
        embedded = self.embedding(input_seq)
        # Remember the input_seq is essentially a batch of text inputs given to the encoder, they all have to be the same length which we already did in our vocab object
        # Packing the data results in computational time being saved, the sequences are already padded and thus now they're packed as a tuple.
        # The tuple contains all the elements in the sequence batch and the other is the batch_size.
        # Refer here on the specific nature on how it reduces computationl time in bi-directioanl GRU's: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch#:~:text=When%20training%20RNN%20(LSTM%20or,8%20sequences%20of%20length%208.
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Each output will utilize the data from the last hidden state (at start it's nothing)
        outputs, hidden = self.gru(packed, hidden)
        # Unpack it.
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]
        return outputs, hidden


# Function to check training loss. Copied.
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = - \
        torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


""""
An attention layer is required to enhance the entire structure, remember in our bidirectional RNN all of the sequence information is passed
from the encoder and then to the decoder via the use through the context vector (its job is to capture the similiar words in the input,
it's calculated by taking the weighted sum of all of the encoder outputs, then it's used for calculating the decoders hidden state which is used
to calcuate the output) but this is a lot of information, thus the attention layer is utilized. In this case we're using the Luoing Attention Layer Implementation
"""


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(
                self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat(
            (hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LEN):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_FLAG for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(
                decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(
                decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            train_loss.append(print_loss_avg)
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(
                encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


teacher_forcing_ratio = 1.0
hidden_size = 500
model_name = 'movie_model'
attention_model = 'dot'
encoder_n_layers = 2
decoder_n_layers = 2
dropoutRate = 0.1
batch_size = 64  # NOTE: MAKE THIS SMALLER IF YOU DON'T HAVE A GPU
train_loss = []
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 250


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(
            1, 1, device=device, dtype=torch.long) * SOS_FLAG
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LEN):
    # Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(
                encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (
                x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# Set the checkpoint to load from; set to None if you're training the model from scratch

#loadFileName = None

checkpoint_iter = 4000
loadFileName = os.path.join(save_dir, model_name, corpus_name,
                            '{}-{}_{}'.format(encoder_n_layers,
                                              decoder_n_layers, hidden_size),
                            '{}_checkpoint.tar'.format(checkpoint_iter))


if loadFileName:  # Check if you're loading from the machine the model was trained on
    checkpoint = torch.load(loadFileName)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFileName:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropoutRate)
decoder = LuongAttnDecoderRNN(
    attention_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropoutRate)
if loadFileName:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(
    decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

if loadFileName:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)


print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFileName)


plt.plot(train_loss)
plt.show()

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

evaluateInput(encoder, decoder, searcher, voc)
