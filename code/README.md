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
 - 0: perp=395.8616,	loss=5.834 	| Val 240.5	5.479
 - 1: perp=182.4724,	loss=5.1968 	| Val 154.021	5.032
 - 2: perp=132.972,	loss=4.8852 	| Val 119.916	4.781
 - 3: perp=107.2805,	loss=4.6717 	| Val 103.109	4.63
 - 4: perp=94.2439,	loss=4.5425 	| Val 93.741	4.534
 - 5: perp=85.7861,	loss=4.4479 	| Val 86.883	4.457
 > Testing: 
 - perp=82.158,	loss=4.4012
 > Total Time: 644.304 sec

With 6 epochs, test perplexity is 86 (<100) and can run under 10 min
