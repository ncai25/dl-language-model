# hw4-s24-template


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
