# Language Model Project: Trigram + RNN

## Overview
Built a Language Model to explore word embedding schemas and minimize NLP losses through word prediction tasks.


## Results

My accuracy/perplexity 

Trigram Model Statistics:
 > Training: 
 - 0: perp=141.6232,	loss=4.6083 	| Val 85.14	4.301
 > Testing: 
 - perp=85.1402,	loss=4.3009
 > Total Time: 122.139 sec

 With only one epoch, test perplexity is 85 (<120)

RNN Model Statistics:
 > Training: 
 - 0: perp=322.6649,	loss=5.0482 	| Val 157.3758
 - 1: perp=127.3712,	loss=4.6884 	| Val 110.0698
 - 2: perp=96.5015,	loss=4.5090 	| Val  92.1912
 - 3: perp=82.6156,	loss=4.4003 	| Val  82.8567 > Testing: 
 - perp=82.158,	loss=4.4012

With 4 epochs, test perplexity is 82.8567 (<100) and can run under 10 min

## Tasks

[Word Prediction] Builted a language model that learns to generate sentences by training the model to predict a next word conditioned on a subset of previous words.

- Trigram: Trained a supervised formulation to predict word[i] = f(word[i-2], word[i-1]).
- RNN: Trained a recurrent neural network to predict word[i] = f(word[0], …, word[i-1]).

[LSTM/GRU] Created a recurrent layer from scratch (with some starter code) using a recurrent kernel.