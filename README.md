# diversity-promoting-dialogue-system

This is a implementation with Pytorch and Python for diversity-promoting dialogue system proposed by Li et al. 2015  

## Requirement:  
Python 3.6.4  
Cuda 8.0  
Pytorch 0.4.0  

## Corpus:
The dataset folder contains a tweets dialogue file round 700k lines. You can also replace with own dataset, as long as the input-output sequence pairs should be in the adjacent lines. For example,  


## Train
To run it, please download or clone this repository. Two jupyter notebooks correspond to MMI-antiLM model and MMI-bidi model.  

When getting started, set train_type as 'restart' in both dataloading part and model initialization part. During training, you can save dataset partition as well as network parameters for future use, by simply set train_type as 'resume'.  
