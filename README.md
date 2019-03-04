# Introduction
This repo contains code of the 5th place solution in [Quora Insincere Question Classification](https://www.kaggle.com/c/quora-insincere-questions-classification/)
The solution is written in Python 3, based on  PyTorch. The experiments are organized by [sacred](https://github.com/IDSIA/sacred).
For details of the solution, check my kernel on [Kaggle](https://www.kaggle.com/jiangm/5th-place-solution).

A modified version of temporal convolutional  network is provided in branch tcn. The f1 is about 0.02 lower than my best rnn model. 
# Usage
Firstly, you need to download data [here](https://www.kaggle.com/c/quora-insincere-questions-classification/data)
and put the data into ./input folder. The parameters can be modified in config.yaml. (comments after the challenge: Better result can be achieved with batch size 768 in the given time limit)  
To start an experiment, run
    
    python expr.py
    
To generate a submission, run 
   
    python script.py
  
To run a hypothesis test, run
    
    python stats.py
    
The result needs to be tracked in mongodb.

 