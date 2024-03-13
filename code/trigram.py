import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace


class MyTrigram(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size=100, embed_size=64):
        """
        The Model class predicts the next words in a sequence.
        : param vocab_size : The number of unique words in the data
        : param hidden_size   : The size of your desired RNN
        : param embed_size : The size of your latent embedding
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.output_layer = tf.keras.layers.Dense(vocab_size, activation='softmax') # tk vocab_size

        ## TODO: define your trainable variables and/or layers here. This should include an
        ## embedding component, and any other variables/layers you require.


    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup or tf.keras.layers.Embedding)
        :param inputs: word ids of shape (batch_size, 2)
        :return: logits: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """
        embedding = self.embedding_layer(inputs)
        embedding = tf.reshape(embedding, (-1, 2*self.embed_size)) # tk 
        #(batch size (-1), words in single input*dimensions for each word)
        logits = self.output_layer(embedding)

        # print(f"logits shape, {tf.shape(logits)}")

        return logits

    def generate_sentence(self, word1, word2, length, vocab):
        """
        Given initial 2 words, print out predicted sentence of targeted length.
        (NOTE: you shouldn't need to make any changes to this function).

        :param word1: string, first word
        :param word2: string, second word
        :param length: int, desired sentence length
        :param vocab: dictionary, word to id mapping

        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}
        output_string = np.zeros((1, length), dtype=np.int32)
        output_string[:, :2] = vocab[word1], vocab[word2]

        for end in range(2, length):
            start = end - 2
            output_string[:, end] = np.argmax(self(output_string[:, start:end]), axis=1)
        text = [reverse_vocab[i] for i in list(output_string[0])]

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
    ## Optional: Feel free to change or add more arguments!
    model = MyTrigram(len(vocab))
    # model = MyTrigram(vocab_size=len(vocab), hidden_size=hidden_size, embed_size=embed_size)

    ## TODO: Define your own loss and metric for your optimizer
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    acc_metric  = perplexity

    # Sanity check for the perplexity calculation
    random_pred = tf.Variable(np.array([[0.1, 0.3, 0.5, 0.1], 
                                        [0.4, 0.3, 0.1, 0.2], 
                                        [0.1, 0.7, 0.1, 0.1], 
                                        [0.3, 0.3, 0.2, 0.2]]))  
    random_true = tf.Variable(np.array([2,0,1,3]))
    np.testing.assert_almost_equal(np.mean(acc_metric(random_true, random_pred)), 2.4446151121745054, decimal=4)

    # print("Passed another Sanity check")
    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)    
    model.compile(
        optimizer=optimizer, 
        loss=loss_metric, 
        metrics=[acc_metric],
    )
    return SimpleNamespace(
        model = model,
        epochs = 1,
        batch_size = 100,
    )



#########################################################################################

def main():
    ## TODO: Pre-process and vectorize the data
    ##   HINT: You might be able to find this somewhere...

    # AUTOGRADER_TRAIN_FILE = '../data/train'
    # AUTOGRADER_TEST_FILE = '../data/test'

    LOCAL_TRAIN_FILE = '/Users/noracai/Documents/CS1470/homework-4p-language-models-norafk/data/train.txt'
    LOCAL_TEST_FILE = '/Users/noracai/Documents/CS1470/homework-4p-language-models-norafk/data/test.txt'
    
    train_id, test_id, vocabulary = get_data(LOCAL_TRAIN_FILE, LOCAL_TEST_FILE)

    ## Process the data
    def process_trigram_data(data):
        X = np.array(data[:-1])
        Y = np.array(data[2:])
        X = np.column_stack((X[:-1], X[1:]))
        return X, Y

    X0, Y0 = process_trigram_data(train_id)
    X1, Y1 = process_trigram_data(test_id)

    vocab = vocabulary

    # Sanity Check!
    assert X0.shape[1] == 2
    assert X1.shape[1] == 2
    assert X0.shape[0] == Y0.shape[0]
    assert X1.shape[0] == Y1.shape[0]

    # print("Passed Sanity Check!")
    # TODO: Implement get_text_model to return the model that you want to use. 

    #- Reshape the input and output data into the Trigram shape
    args = get_text_model(vocab)
    # args = get_text_model(vocab, hidden_size=150, embed_size=128)
    # print("X0 shape:", X0.shape)
    # print("args.batch_size:", args.batch_size)

    args.model.fit(
        X0, Y0,
        epochs=args.epochs, 
        batch_size=args.batch_size,
        validation_data=(X1, Y1)
    )

    ## Feel free to mess around with the word list to see the model try to generate sentences
    words = 'speak to this brown deep learning student'.split()
    for word1, word2 in zip(words[:-1], words[1:]):
        if word1 not in vocab: print(f"{word1} not in vocabulary")
        if word2 not in vocab: print(f"{word2} not in vocabulary")
        else: args.model.generate_sentence(word1, word2, 20, vocab)

if __name__ == '__main__':
    main()
