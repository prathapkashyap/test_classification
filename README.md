### Creating embeddings

Embeddings are created using Bert and GPT 2 models. 
The embeddings craeted in the project do not reflect the entire dataset. Only around 5000 of the initial documents are used to create embeddings.

To create embeddings and to pass a predered model, run the command 
`python main.py --model bert` to use Bert model to create the embeddings
`python main.py --model gpt-2` to use gpt-2 to create the embeddings

These models are used to tokenize the 20 news group dataset data and the craeted embeddings are saved in a file to be used for training the model

### Training the classifier 
The model is present in the src/models folder. 
Only one simple model is done, a simple linear model.

The model is written using the standards used for pytorch development. Because of some issues in input sizes and matrix multiplication, the model was not trained. I will try more to correct the error and push it once it is fixed.

To train the model, while being present in the models directory, run
`python main.py --model bert` or `python main.py --model gpt-2` to use the embeddings based on the name of the model.

I have written the code for training and testing the model as well, but because of the error it could not be tested.
