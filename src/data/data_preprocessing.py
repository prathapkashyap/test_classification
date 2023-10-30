from src.data.data_loader import fetch_training_data, fetch_testing_data
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2Model
import torch
import numpy as np
from src.utils.constants import model_names

def get_tokenizer(model_name:str):
    print("5. print tokenizer and 8")
    if(model_name == model_names.get('bert')):
        return AutoTokenizer.from_pretrained("bert-base-uncased")
    elif(model_name == 'gpt-2'):
        return GPT2Tokenizer.from_pretrained('gpt2')

def get_model(model_name:str):
    if(model_name == 'bert'):
        return AutoModel.from_pretrained("bert-base-uncased")
    if(model_name == 'gpt-2'):
        return GPT2Model.from_pretrained('gpt2')

def save_embeddings_to_file(model_name:str, embeddings:list, file_name: str):
    embeddings_array = np.array(embeddings)
    save_path = file_name+".npy"
    np.save(save_path, embeddings_array)
    embeddings = [embedding.tolist() for embedding in embeddings]
    save_path = file_name+".txt"
    with open(save_path, 'w') as file:
        for embedding in embeddings:
            line = ','.join(map(str, embedding)) + '\n'
            file.write(line)
        

def create_embeddings(model_name, documents):
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)
    print("model name", model_name, "model:", model)
    max_length = 512
    embeddings = []
    for i,doc in enumerate(documents):
        if(model_name == "bert"):
            tokens = tokenizer.tokenize(doc)[:max_length-2]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
            input_ids = torch.tensor(input_ids).unsqueeze(0)

            
        elif(model_name == "gpt-2"):
            truncated_doc = doc[:1024]
            input_ids = tokenizer.encode(truncated_doc, add_special_tokens=True, return_tensors="pt")

        with torch.no_grad():
            output = model(input_ids)
            embeddings.append(output.last_hidden_state.mean(dim =1).numpy())
    
    return embeddings



def pre_process_data(model_name: str):
    # fetch data
    print("2. Pre process function call")
    documents = fetch_training_data()
    test_documents = fetch_testing_data()
    # print(type(documents), documents["training_data"][0], documents["target"][0])

    train_embeddings = create_embeddings(model_name=model_name, documents=documents["training_data"][:100])
    test_embeddings = create_embeddings(model_name=model_name, documents = test_documents["testing_data"][:10])
    

    # write embeddings to file
    print("embeddings length", len(train_embeddings))
    save_embeddings_to_file(model_name = model_name, embeddings = train_embeddings, file_name = model_name+"_train_embeddings")
    save_embeddings_to_file(model_name = model_name, embeddings = test_embeddings, file_name = model_name+"_test_embeddings")