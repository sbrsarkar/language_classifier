# language_classifier
Identify the language of a given word. Used a character level RNN to predict the language. 
Each character is first embedded into a vector space then the sequence of vectors is passed into two Gated Recurrent Units (GRU) [1]. Finally a fully connected neural retwork with softmax activation function predicts the language.

## predict
run the following command in terminal:
```
python predict.py 'thanks' 
```
<img src="https://github.com/sbrsarkar/language_classifier/blob/master/data/predict.png" alt="sample outputs" width="450" height="350">

## model
<img src="https://github.com/sbrsarkar/language_classifier/blob/master/predict.png" alt="model_architecture" width="450" height="350">


## references
[1]: Cho, Kyunghyun, Bart Van Merriënboer, Dzmitry Bahdanau, and Yoshua Bengio. "On the properties of neural machine translation: Encoder-decoder approaches." arXiv preprint arXiv:1409.1259 (2014).
