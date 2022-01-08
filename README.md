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
In this part, we do text processing at character level. First, we need to find all the distinct character and build connection between character and id.In other words, it is create vocabulary. Here shows the distinct characters. We then do some transform of our text and represent it as a vector of numbers.Those number is composed by the id we assigned to each character.Our documents have overall 357095 characters.

