# Text-generation-with-O-henry-s-short-stories

## 1. Task
Our task is to build a model to generate text. Basically, it uses language model to do that.A language model is simply a probability distribution over a sequence of words. That is a condition probability distribution.Broadly speaking,a statistical language model is a probability distribution over sequences of tokens.Given a prompt, say of length m, a trained Language Model will assign a conditional probability distribution over the dictionary (vocabulary) tokens.We can use the conditional probability distribution to select (sample) the next token to complete the given prompt.
For example, when the prompt is “I want to cook”, the trained language model can output the probability of each token in the dictionary to be the next token as below.Then, according to the implemented sampling method, one can pick the next token according to this probability distribution.

## 2. Data
### 2.1 Sixes and sevenes

In this project, we choose to load the short stories of O.Henry. The file is a text file that you can download from the data folder on my github:
Click on this link : https://github.com/jinghua-duan/Text-generation-with-O-henry-s-short-stories/blob/main/sixes%20and%20sevenes%20ohenry.txt
There are 25 short stories in this text file. Our documents have overall 357095 characters.

### 2.2 Text processing 
In this part, we do text processing at character level. First, we need to find all the distinct character and build connection between character and id.In other words, it is create vocabulary. Here shows the distinct characters. We then do some transform of our text and represent it as a vector of numbers.Those number is composed by the id we assigned to each character.Our documents have overall 357095 characters. After that, we create the input and output of the model. Since our step size is 1. We have 356995 sequences of characters,each sequence has 100 characters. And the depth of each character is 1. It is exactly character level text representation.
In our model, we predict Y character by character using X. We have 356995 characters in Y and we trans form each character into dummy, then we will have 356995 ✖️ 81 dimension of Y.

## 3. Model
### 3.1 Model 1
In this part, I choose to use LSTM , a classical model for text generation. I don't want to talk too much about the principle given there are too many
passages about it. For practical purpose, I will give the procedure for building the model. As we already prepared the dataset, we need to define the input accordingly. Remember that in the train set, the length of the input (X) sequence (sequence_length) is 100 tokens (chars). Thus our model's input is 100✖️1, that means every time we feed our model 100 characters and it gives the output of next character which is a categorical variable.

We use sequential model here.The following are our model.

Model: "sequential"

_________________________________________________________________

 Layer (type)                Output Shape              Param # 
 
=================================================================

 lstm (LSTM)                 (None, 100, 400)          643200   
                                                                 
 dropout (Dropout)           (None, 100, 400)          0        
                                                                 
 lstm_1 (LSTM)               (None, 400)               1281600   
                                                                 
 dropout_1 (Dropout)         (None, 400)               0         
                                                                 
 dense (Dense)               (None, 81)                32481     
                                                                 
=================================================================

Total params: 1,957,281

Trainable params: 1,957,281

Non-trainable params: 0

_________________________________________________________________

None

### 3.2 Model 1 results

Then we start to fit the model. 

Epoch 1/2
175/175 [==============================] - ETA: 0s - loss: 3.1518 
Epoch 00001: loss improved from inf to 3.15183, saving model to /content/baseline-improvement-ohenry-01-3.1518.hdf5
175/175 [==============================] - 9052s 52s/step - loss: 3.1518
Epoch 2/2
175/175 [==============================] - ETA: 0s - loss: 2.9086 
Epoch 00002: loss improved from 3.15183 to 2.90858, saving model to /content/baseline-improvement-ohenry-02-2.9086.hdf5
175/175 [==============================] - 9243s 53s/step - loss: 2.9086
<keras.callbacks.History at 0x7ff6b60f4850>

From above, we can see our model performs poorly. The loss is still large and is very computationally expensive. I spend 8 hours to train this model. 
Let us see the results of text generation.Here we select some random sequence of our text with length of sequence as 100 characters. And use it to generate the text.We just plug in our 100 characters into our model to predict next characters and do it repeatedly until obtaining plausible sequences.

Seed:
" seats. Two of them
made a fight and were both killed.

It took the Daltons just ten minutes to captu "

Results:
" seats. Two of them
made a fight and were both killed.

It took the Daltons just ten minutes to captu to the toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe toe"

Clearly, this model failed. From the results, we can see that our model only generate "toe" repeatedly. It makes no sense. Why our model performs so baddly?one reason is our model is underfitting. Another reason is we didn't do sampling when generating text. Or it is because we didn't do text cleaning.

### 3.3 Model 2

In this part, we will do something different from part 1. We use tensorflow and keras to process the data. First, we concatenate all the text in one line. Thus we can escape from separator and other weird notation.We have 80 distinct characters and 350160 overall characters in our text. This is because we remove delimiter.Next we will change the construction of our character sequences.First, we set a shorter character sequence length from 100 to 20.Second, we create next sequence of characters 3 steps after rather than 1 step after.Here we choose to use tensorflow to vectorize our data, rather than directly use the id constructed by default.The batch size is 64,thus the input and output dimension is therefore 64✖️20 and 64✖️1. 

Thus, we provide this info to the embedding layer.Next, we add a layer to map those vocab indices into a space of dimensionality ‘embedding_dim’.After applying Dropout, we use an LSTM layer to process the sequence and learn to generate the next token with the help of a Dense layer. Since we use scalar to represent the output (Y), that is; we do not use one-hot encoding, we need to use sparse_categorical_crossentropy loss function.

h
Thus, we provide this info to the embedding layer.Next, we add a layer to map those vocab indices into a space of dimensionality ‘embedding_dim’.After applying Dropout, we use an LSTM layer to process the sequence and learn to generate the next token with the help of a Dense layer. Since we use scalar to represent the output (Y), that is; we do not use one-hot encoding, we need to use sparse_categorical_crossentropy loss function.

Here are the models.

Model: "model_LSTM"

_________________________________________________________________

 Layer (type)                Output Shape              Param #   
 
=================================================================

 input_1 (InputLayer)        [(None, 20)]              0         
                                                                 
 embedding (Embedding)       (None, 20, 16)            1536      
                                                                 
 dropout (Dropout)           (None, 20, 16)            0         
                                                                 
 lstm (LSTM)                 (None, 20, 512)           1083392   
                                                                 
 flatten (Flatten)           (None, 10240)             0         
                                                                 
 dense (Dense)               (None, 96)                983136    
                                                                 
=================================================================

Total params: 2,068,064

Trainable params: 2,068,064

Non-trainable params: 0
_________________________________________________________________

None

### 3.4 Model 2 results

Before generate text, we need define some sampling method to draw sample from the conditional distribution. Why do we need sampling?
Because sampling means randomly picking the next word according to its conditional probability distribution. After generating a probability distribution over vocabulary for the given input sequence, we need to carefully decide how to select the next token from this distribution. There are several methods to do that. Like greeedy search temperature and Top-k sampling.

Let us see the results:

The prompt is

	 he talked to her in 
   
Text Generated by Greedy Search Sampling:
	 he talked to her in soon for the have of a kinge to a maln coruing out betsime that pieces to had frong edpotiency unt
Text Generated by Temperature Sampling:
	temperature:  0.2
	 he talked to her in soon of garest sworld that to me about to a man ffom the molne the capperon a sprank hime and wi
	temperature:  0.5
	 he talked to her in sore doyty wall      of the salinote hoase worside hed noted my lace awas in his rand
	temperature:  1.0
	 he talked to her in dyor for day ecsumllr with a man from the weme a witch gleat now and that badinewimls of to
	temperature:  1.2
	 he talked to her in sheers pkpsive instlads or peh from the seen me wehe and redecen entorated of half never    i
Text Generated by Top-K Sampling:
	Top-k:  2
	 he talked to her in shewimg sowsly hape that the yabee of a hepertyrar nwert i heard of pyemiled a little sir not an
	Top-k:  3
	 he talked to her in somolar on busiellsias wile so fiss ralge ones fartuoosus hadientar poptresis or roudtand  al
	Top-k:  4
	 he talked to her in danwer pin i coulddlve mectednamalkiplyssekfto fise  m aldneloow in gaibs w aped celled trach
	Top-k:  5
	 he talked to her in syoun oot svalkerr fof fovee exmacl ouct bes tiles   ge trinvly das obt thybolk triingert 

## 4. Conclusion

From above, we can see that our model achieve a relatively small loss, high accuracy and very fast training speed. This is because we use tensor flow to processing data. Besides, we do not use sequential model.And it can produce some meaningful text, not like the first model. Besides, we can also use different sampling method to generate text.

## 5. Reference

https://github.com/DLProjectTextGeneration/TextGeneration




