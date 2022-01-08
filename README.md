# Text-generation-with-O-henry-s-short-stories

## 1. Task
Our task is to build a model to generate text. Basically, it uses language model to do that.A language model is simply a probability distribution over a sequence of words. That is a condition probability distribution.Broadly speaking,a statistical language model is a probability distribution over sequences of tokens.Given a prompt, say of length m, a trained Language Model will assign a conditional probability distribution over the dictionary (vocabulary) tokens.We can use the conditional probability distribution to select (sample) the next token to complete the given prompt.
For example, when the prompt is “I want to cook”, the trained language model can output the probability of each token in the dictionary to be the next token as below.Then, according to the implemented sampling method, one can pick the next token according to this probability distribution.
