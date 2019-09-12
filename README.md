# RecMF
+ This is a implement of matrix factorization based Recommendation System. 
+ In order to find out the distribution of user behaviors, We quantized the data by seting levels for difference times of users contact items, seting of quantization is listed below.
+ Learning the distribution of user behaviors by applied bidirectional LSTM.
```python 
#  0 ---- 1 ---- 2 ---- 3 ---- 4 ---- 5 ---- =>
#  |  q0  |  q1  |  q2  |  q3  |  q4  |  q5  =>
quantize_unit = 1
max_quantize = 10
```
## Data Preprocess
### Dataset
+ jester 2
### Data format
+ Data of training, testing and validation should be save as an users-to-items matrix respectively as following:
Users v.s. Items | Item 1 | Item 2 | ... | Item N
:---------------:|:------:|:------:|:---:|:-----:
User 1           |Score_11|Score_12| ... |Score_1N 
User 2           |Score_21|Score_22| ... |Score_2N
  :              |   :    |   :    | ... |   :
User M           |Score_N1|Score_N2| ... |Score_NN
### Input Format
>     |<--  one-hot of users  -->|<-  one-hot of items  ->|
> x = [0 0 0 0 0 ... 0 1 0 ...  0 0 ... 0 1 0 ...0 0 0 0 0]
+ Input data which means user_i has watched item_j are concatenation of one-hot of users and items. 

## Train
+ Architecture of model is defined in "RecMF/src/rec_mf.py"
+ Training code can be found in "RecMF/src/train_model.py"
+ Put your dataset into the directory:"RecMF/Dataset/"
