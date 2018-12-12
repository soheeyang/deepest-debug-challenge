# Deepest Debug Challenge: can you squash them all?

WARNING: This model is implemented to NOT work correcly. DO NOT use this for your application. It will hurt you.

This is a quiz code for debugging a badly-implemented neural network.

The code runs fine and trains the model somehow, but not correctly.

You are given a task to fix all the bugs in the code.

__This code is for Quest 2 of [Deepest Season 5 Recruiting](https://drive.google.com/file/d/14nG3DwQIBcWFgD9YmOKYa__8HEYOZMK7/view).__

## Updates

- __12/05__
It seems like the original description in the link above was not clear enough.  
Therefore, I rewrote some of the the descriptions just to make the requirements more clear.  
Please refer to this link: __http://bit.ly/2ANdfMW__
- __12/12__
If you fix all the bugs, only __one epoch of training__ would be enough to get an accuracy of around 80%. Therefore, those who do not have any GPU, please feel free to tackle this problem. (In fact, I myself did not use any GPU till I finished designing this quest.)

# Usage

Use python >= 3.6.

1. run `python prepro.py` to make `data/prepro.npz`, which is the data used to train and test the model.

2. run `python train.py` to train and test the model.
