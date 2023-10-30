from Text_Style_Embeddings_from_Topic_Models.src.utils.utils import parse_arguments
from model import Classifier
from data.data_loader import fetch_target_names, fetch_training_labels, fetch_test_labels
from torch import nn
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path


def read_embeddings(file_name):

    saved_embeddings = np.load(file_name)
    return saved_embeddings


def create_data_loader(file_name, labels):
    inputs = torch.tensor(read_embeddings(file_name))
    labels_tensors  = torch.tensor(labels[:len(inputs)])
    dataset = TensorDataset(inputs, labels_tensors)
    batch_size = 32
    print(f"inputs_______:{inputs.shape}")
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle= True)
    return train_loader


def train(embeddings_file_name: str, 
        train_model: torch.nn.Module,  
        # data_loader: torch.utils.data.DataLoader,
        loss_fn : torch.nn.Module,
        optimizer: torch.optim.Optimizer):
    print("train function")    
    epochs = 10

    train_loader = create_data_loader(file_name=embeddings_file_name, labels= fetch_training_labels())
    for epoch in range(epochs):
        train_model.train() 
        for input, label in train_loader:
            y_pred_class = train_model(input)
            loss = loss_fn(y_pred_class, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_class = torch.argmax(torch.softmax(y_pred_class, dim = 1), dim = 1)
            train_acc += (y_pred_class ==label).sum().item()/len(y_pred_class)
    train_loss = train_loss/len(train_loader)
    train_acc = train_acc/len(train_loader)
    return train_loss, train_acc

def test(model_name: str, 
        test_model: torch.nn.Module,  
        loss_fn : torch.nn.Module):
    test_model.eval()
    test_loss, test_acc = 0, 0
    test_data_loader = create_data_loader(model_name=model_name, labels= fetch_test_labels())
    with torch.inference_mode():
        for input, label in test_data_loader:
            test_y_logits = test_model(input)
            loss = loss_fn(test_y_logits, label)
            test_loss+= loss.item()

            test_pred_labels = test_y_logits.argmax(dim =1)
            test_acc += ((test_pred_labels == label).sum().item()/len(test_pred_labels))
    
    test_loss = test_loss/len(test_data_loader)
    test_acc = test_acc/len(test_data_loader)
    return test_loss, test_acc

def save_model(model:str, model_file_name: str):
    torch.save(obj = model.state_dict(), f = model_file_name)
    # model_path = Path("models")
    # model_name = "text_classifier_bert.pth"

def main(model_name: str):
    print("Main function, train")
    target_classes = fetch_target_names()
    num_classes = len(target_classes)
    model = Classifier(num_classes= num_classes, embedding_size = 512)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    [train_loss, train_acc] = train(embeddings_file_name=model_name+"_train_embeddings.npy", 
                                train_model=model,
                                loss_fn = loss,
                                optimizer = optimizer)
    save_model(model= model, model_file_name="classifier_trained_with_"+model_name+"_embeddings.pth" )
    

    print(f"train_loss: {train_loss}, train_acc:{train_acc}")

if __name__ == "__main__":
    args = parse_arguments()
    model_embeddings_to_use = args.model
    main(model_name=model_embeddings_to_use)
