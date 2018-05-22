# diversity-promoting-dialogue-system

This is a implementation with Pytorch and Python for [diversity-promoting dialogue system](https://arxiv.org/pdf/1510.03055.pdf) proposed by Li et al. 2015  


## Requirement:  
Python 3.6.4 (not limited to)  
Jupyter Notebook  
Cuda 8.0  
Pytorch 0.4.0  


## Corpus:
The dataset folder contains a tweets dialogue file round 700k lines. You can also replace with own dataset, as long as the input-output sequence pairs are in the adjacent lines. For example,  
```
message1
response1
message2
response2
```


## Train:
Two jupyter notebooks correspond to MMI-antiLM model and MMI-bidi model. To run it, please download or clone this repository.  

When getting started, set train_type variable as 'restart' in dataloading part (block No.5). During training, you can save dataset partition as well as network parameters for future use, by simply setting train_type as 'resume'.  


## Evaluation:
Beam search is built for decoding, and the main measurements are BLEU score and distinct value of unigrams and bigrams. In notebook, there are some records with different hyper-parameters I tried. One can also play with other combination.  
For MMI-antiLM, evaluation on test set achieves BLEU 4.21, with 0.126 and 0.457 for distinct-1 and distinct-2.  
For MMI-bidi, evaluation on test set achieves BLEU 4.20, with 0.111 and 0.408 for distinct-1 and distinct-2.  

## Result (sample):
Here are a few results sampled from test set.

MMI-antiLM -- gamma=0.2, lambda=0.2, threshold=2
![sample](https://github.com/YifanZhou95/diversity-promoting-dialogue-system/blob/master/sample/sample_retweet_anti.PNG)

MMI-bidi -- gamma=-0.2, lambda=0.5
![sample](https://github.com/YifanZhou95/diversity-promoting-dialogue-system/blob/master/sample/sample_show_bidi.PNG)

