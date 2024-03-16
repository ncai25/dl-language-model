import tensorflow as tf
import numpy as np
import random
from preprocess import get_data
from types import SimpleNamespace


class MyRNN(tf.keras.Model):

    ##########################################################################################

    def __init__(self, vocab_size, rnn_size=128, embed_size=64):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param rnn_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size


        ## TODO:
        ## - Define an embedding component to embed the word indices into a trainable embedding space.
        ## - Define a recurrent component to reason with the sequence of data. 
        ## - You may also want a dense layer near the end...    
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn_layer = tf.keras.layers.LSTM(rnn_size, return_sequences=True, return_state=False)
        # return sequence because we need timesteps - why
        # `timesteps` is basically our `window`. - we treat a sequence of words as a time-series data.
        
        # # return_state= True, last hidden layer and last cell state 
        # #  return_sequences = True, whole sequence of outputs. shape [batch, timesteps, embedding]

        self.output_layer = tf.keras.layers.Dense(vocab_size, activation='softmax') 


    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup or tf.keras.layers.Embedding)
        - You must use an LSTM or GRU as the next layer.
        """
        X_RNN_embedding = self.embedding_layer(inputs)
        rnn_output = self.rnn_layer(X_RNN_embedding) # we don't need to reshape here unlike trigram tk
        logits = self.output_layer(rnn_output)

        # print(f'embedding shape, {X_RNN_embedding.shape}')
        # print(f'rnn_output shape, {rnn_output.shape}')
        # [batch, timesteps, rnn_size]

        return logits
    
        # X_RNN_embedding = tf.reshape(X_RNN_embedding, (-1, 2*self.embed_size)) # bc embedding in trigram is like 2 colomns

        # self.rnn_layer.build(X_RNN_embedding)
        # lstm_weights = self.rnn_layer.get_weights()
        # self.rnn_layer.set_weights(lstm_weights)

    ##########################################################################################

    def generate_sentence(self, word1, length, vocab, sample_n=10):
        """
        Takes a model, vocab, selects from the most likely next word from the model's distribution.
        (NOTE: you shouldn't need to make any changes to this function).
        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}

        first_string = word1
        first_word_index = vocab[word1]
        next_input = np.array([[first_word_index]])
        text = [first_string]

        for i in range(length):
            logits = self.call(next_input)
            logits = np.array(logits[0,0,:])
            top_n = np.argsort(logits)[-sample_n:]
            n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
            out_index = np.random.choice(top_n,p=n_logits)

            text.append(reverse_vocab[out_index])
            next_input = np.array([[out_index]])

        print(" ".join(text))


#########################################################################################
def perplexity(y_true, y_pred):
    sparse_cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    avg_cross_entropy = tf.reduce_mean(sparse_cross_entropy)
    return tf.exp(avg_cross_entropy)

def get_text_model(vocab):
    '''
    Tell our autograder how to train and test your model!
    '''
    ## TODO: Set up your implementation of the RNN
    ## Optional: Feel free to change or add more arguments!
    model = MyRNN(len(vocab))

    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    acc_metric  = perplexity

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)    

    model.compile(
        optimizer=optimizer, 
        loss=loss_metric, 
        metrics=[acc_metric],
    )

    return SimpleNamespace(
        model = model,
        epochs = 4,
        batch_size = 50,
    )



#########################################################################################

def main(): 

    ## TODO: Pre-process and vectorize the data
    ##   HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    ##   from train_x and test_x. You also need to drop the first element from train_y and test_y.
    ##   If you don't do this, you will see very, very small perplexities.
    ##   HINT: You might be able to find this somewhere...
    LOCAL_TRAIN_FILE = '/Users/noracai/Documents/CS1470/homework-4p-language-models-norafk/data/train.txt'
    LOCAL_TEST_FILE = '/Users/noracai/Documents/CS1470/homework-4p-language-models-norafk/data/test.txt'
    
    train_id, test_id, vocabulary = get_data(LOCAL_TRAIN_FILE, LOCAL_TEST_FILE)
   
    window_size = 20 
    offset = random.randint(0, window_size - 1) # randomly pick one value
    # print(offset)

# - In practice we draw a **random integer between 0 and the (window size-1)** for every epoch
# - Then we remove the random integer number of words from the beginning of the training corpus (= offset)
# - Depending on the value of the random integer,
#   - Different words end up not being included in `X` and `y` from one epoch to another
#   - `X` and `y` end up having different number of windows from one epoch to another
#   - For every epoch, the model is trained with similar but slightly different sets of windows 
    

# # X_RNN_embedding = tf.reshape(X_RNN_embedding, (-1, 2*self.embed_size)) # tk 
    # didn't reshape in call but here

    train_id = np.array(train_id)
    test_id  = np.array(test_id)

    def process_rnn_data(data):
        offset_data = (data[offset:])
        remainder = (len(offset_data)-1)%window_size
        X = tf.reshape(offset_data[:-remainder][:-1], (-1, window_size)) # remove the last elm
        Y = tf.reshape(offset_data[:-remainder][1:], (-1, window_size)) # drop first elm

        return X, Y

    X0, Y0 = process_rnn_data(train_id)
    X1, Y1 = process_rnn_data(test_id)

    # print(f"X_RNN shape = {X0.shape}")
    # print(f"Y_RNN shape = {Y0.shape}")

    vocab = vocabulary

    ## TODO: Get your model that you'd like to use
    args = get_text_model(vocab)

    args.model.fit(
        X0, Y0,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(X1, Y1)
    )

    ## Feel free to mess around with the word list to see the model try to generate sentences
    for word1 in 'speak to this brown deep learning student'.split():
        if word1 not in vocab: print(f"{word1} not in vocabulary")            
        else: args.model.generate_sentence(word1, 20, vocab, 10)

if __name__ == '__main__':
    main()
