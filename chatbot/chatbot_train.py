# Class For Encoder Recurrent Neural Network
# Utilize base class for neural network modules (torch) refer here: https://pytorch.org/docs/stable/generated/torch.nn.Module.html


# Understand GRU's: https://www.analyticsvidhya.com/blog/2021/03/introduction-to-gated-recurrent-unit-gru/

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import itertools

from text_processing import SOS_FLAG, EOS_FLAG, PAD_FLAG, MAX_LEN, voc, corpus_name, pairs, save_dir
from torch import optim

CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")


class EncoderRNN(nn.Module):

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        # Allows us to reference EncoderRNN obj with the use of indirection. Refer here: https://www.programiz.com/python-programming/methods/built-in/super
        # The Encoder Mechanism is defined in a separate class
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers  # Denote number of layers
        self.hidden_size = hidden_size  # Denote hidden layer size
        # Denote embedding layer (in this case we're not utilzing one-hot encoding due to the massive size of our vocabulary)
        self.embedding = embedding
        # Load up Pytorch GRU module from NN. Gated Recurrent unit is an alternate version of an LSTM (Long Short Term Memory RNN)
        # Think of them as simplified logic gates to save and throw away input data. We set the birectional flag to true.
        # Thus a birectional GRU is being used which utilizes the input sequence which is reverse and passed onto another GRU
        # Refer here: https://blog.paperspace.com/bidirectional-rnn-keras/
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(
            0 if n_layers == 1 else dropout), bidirectional=True)

    #
    def forward(self, input_seq, input_length, hidden=None):
        # We first must convert the text input and embed it
        embedded = self.embedding(input_seq)
        # Remember the input_seq is essentially a batch of text inputs given to the encoder, they all have to be the same length which we already did in our vocab object
        # Packing the data results in computational time being saved, the sequences are already padded and thus now they're packed as a tuple.
        # The tuple contains all the elements in the sequence batch and the other is the batch_size.
        # Refer here on the specific nature on how it reduces computationl time in bi-directioanl GRU's: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch#:~:text=When%20training%20RNN%20(LSTM%20or,8%20sequences%20of%20length%208.
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, input_length)
        # Each output will utilize the data from the last hidden state (at start it's nothing)
        outputs, hidden = self.gru(packed, hidden)
        # Unpack the padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum GRU output
        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]

        return outputs, hidden


""""
An attention layer is required to enhance the entire structure, remember in our bidirectional RNN all of the sequence information is passed 
from the encoder and then to the decoder via the use through the context vector (its job is to capture the similiar words in the input,
it's calculated by taking the weighted sum of all of the encoder outputs, then it's used for calculating the decoders hidden state which is used
to calcuate the output) but this is a lot of information, thus the attention layer is utilized. In this case we're using the Luoing Attention Layer Implementation
"""


class AttentionLayer(torch.nn.Module):  # USING LUONG ATTENTION IMPLEMENTATION

    def __init__(self, method, hidden_size):
        # The Attention Mechanism is defined in a separate class
        super(AttentionLayer, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(
                self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.AttentionLayer = torch.nn.Linear(
                self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.AttentionLayer = torch.nn.Linear(
                self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_ouput):
        energy = self.AttentionLayer(encoder_ouput)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.AttentionLayer(torch.cat(
            (hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()

        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttentionDecoder(nn.Module):
    def __init__(self, attention_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttentionDecoder, self).__init__()

        self.attention_model = attention_model
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

        self.attn = AttentionLayer(attention_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Get embedding of current input word (Remember this would be the N dimensional point)
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
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
        # Predict next word using Luong
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


# Function to check training loss. Copied.
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = - \
        torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LEN):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS flags for each sentence)
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
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_FLAG]


def zeroPadding(l, fillvalue=PAD_FLAG):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


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

# Returns padded input sequence tensor and lengths


def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length


def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs


def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        # Start from the iteration of the checkpoint
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

        # It's possible to save a checkpoint of the point you've trained to with PyTorch.
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
hidden_size = 512
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
n_iteration = 500
print_every = 10
save_every = 250

# Set the checkpoint to load from; set to None if you're training the model from scratch
loadFileName = None
#checkpoint_iter = 200000
# loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                           '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                             '{}_checkpoint.tar'.format(checkpoint_iter))

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
decoder = LuongAttentionDecoder(
    attention_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropoutRate)
if loadFileName:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

encoder.train()
decoder.train()

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
