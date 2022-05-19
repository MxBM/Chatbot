import torch
import torch.nn as nn
import os
from io import open
import pickle
from flask import Flask, abort, request
import json
import sqlite3
from sqlite3 import Error
import torch.nn.functional as F

import unicodedata
import re


attention_model = 'dot'


PAD_FLAG = 0  # Padding for shorter sentences
SOS_FLAG = 1  # Start of sentence
EOS_FLAG = 2  # End of sentence
MAX_LEN = 10
MIN_COUNT = 3  # Minimum word count for trimming.
encoder_n_layers = 2  # Number of layers in the encoder
decoder_n_layers = 2  # Number of layers in the decoder
hidden_size = 500
dropoutRate = 0.1
loadFileName = "O:/Chatbot/model/save/movie_model/chatbot/2-2_500/2000_checkpoint.tar"


CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")


def connect(db_file):  # Copied function, utilized to connect to database that maintains
    try:
        connection = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    finally:
        connection.close()


app = Flask(__name__)  # Pass in current module ,chatbot, as app


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


# Open JSON file of contractions in read mode.
with open('O:/Chatbot/chatbot/contractions.json', 'r') as fp:
    fixed_prefix = json.load(fp)  # Get JSON file and return it as an object
    fixed_prefix = {key.strip().lower(): value.strip().lower()
                    for key, value in fixed_prefix.items()}


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_FLAG]


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


@app.route('/chatbot', methods=['POST'])
def foo():
    if not request.json:
        abort(400)
    input_sentence = " ".join([each_word for each_word in normalizeString(
        request.get_json()["message"]).split() if each_word in set_voc])
    # Evaluate sentence
    output_words = evaluate(encoder, decoder, searcher, voc,
                            input_sentence, max_length=len(input_sentence.split()))
    # Format and print response sentence
    output_words[:] = [x for x in output_words if not (
        x == 'EOS' or x == 'PAD')]
    output_words = " ".join(output_words)
    try:
        conn = sqlite3.connect("datastore.db")
        conn.execute('''CREATE TABLE qanda(
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Question           TEXT      NOT NULL,
        Answer            TEXT       NOT NULL);''')
        print("Table created successfully")
        conn.close()
    except:
        conn.close()
    try:
        conn = sqlite3.connect("datastore.db")
        conn.execute('''INSERT INTO qanda (Question,Answer)
        VALUES ( ?, ?)''', input_sentence, output_words)
        conn.commit()
        print("Value inserted successfully")
        conn.close()
    except:
        conn.close()

    return json.dumps({
        "text": output_words
    })


print(__name__)

if __name__ == '__main__':
    #######################################################all the important functionalities#######################################################
    connect("datastore.db")

    # voc used for both utils and app
    with open("O:/Chatbot/chatbot/pickles/vocab.file", "rb") as f:
        voc = pickle.load(f)
    set_voc = set(voc.word2index.keys())

    # Load model if a loadFileName is provided
    if loadFileName:
        print('loading the file')
        # If loading on same machine the model was trained on
        # checkpoint = torch.load(loadFileName)
        # If loading a model trained on GPU to CPU
        checkpoint = torch.load(loadFileName, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
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

    # Set dropoutRate layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)
    #######################################################all the important functionalities#######################################################
    app.run(host='0.0.0.0', port=5000, debug=True)
    os.system("cd .. & npm start")
