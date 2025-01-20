import pandas as pd
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset

embedding_directory = 'embeddings/GCB_Pooler_embeddings_patch_before_largewithcodedata_15300limit3label'
model_directory = 'models/two_pipelines_StScaler_pooler_6layers_invertednn_100epoch_batch_normalization.pth'
device = 'cuda:0'

def onehot_embed_df(df):
    # One-hot encoding ile labels sütununu dönüştürme
    df_encoded_labels = pd.get_dummies(df, columns=['labels'])
    df_encoded_repo = pd.get_dummies(df, columns=['full_repo_name'])

    # Özellikler ve etiketler
    X_repo = df_encoded_repo.drop(columns=['labels','patch','index','file_path'])
    X_repo = X_repo.astype(int)

    # y = df_encoded.drop(columns=['content_before', 'content_after', 'tokens'])
    y = df_encoded_labels.drop(columns=['patch', 'full_repo_name','index','file_path'])
    y = y.astype(int) # convert it 0,1     instead of true, false 

    df['label_tensor'] =[torch.tensor(row,dtype=torch.float32) for row in y.values] 
    df['repo_name_tensor'] =[torch.tensor(row,dtype=torch.float32) for row in X_repo.values] 

    return df.drop(columns=['labels','patch'])


# Define the ANN model
class CodeDiffNN(nn.Module):
    def __init__(self, code_input_size, repo_input_size, hidden_sizes_embedding_pipeline,hidden_sizes_reponame_pipeline,hidden_sizes_merged_pipeline, output_size, dropout_rate):
        super(CodeDiffNN, self).__init__()
        
        # Pipeline for code embeddings
        self.code_fc1 = nn.Linear(code_input_size, hidden_sizes_embedding_pipeline[0])
        self.code_bn1 = nn.BatchNorm1d(hidden_sizes_embedding_pipeline[0])
        self.code_relu1 = nn.ReLU()
        self.code_dropout1 = nn.Dropout(dropout_rate)
        
        self.code_fc2 = nn.Linear(hidden_sizes_embedding_pipeline[0], hidden_sizes_embedding_pipeline[1])
        self.code_bn2 = nn.BatchNorm1d(hidden_sizes_embedding_pipeline[1])
        self.code_relu2 = nn.ReLU()
        self.code_dropout2 = nn.Dropout(dropout_rate)
        
        self.code_fc3 = nn.Linear(hidden_sizes_embedding_pipeline[1], hidden_sizes_embedding_pipeline[2])
        self.code_bn3 = nn.BatchNorm1d(hidden_sizes_embedding_pipeline[2])
        self.code_relu3 = nn.ReLU()
        self.code_dropout3 = nn.Dropout(dropout_rate)  
        
        
        # Pipeline for repo names
        self.repo_fc1 = nn.Linear(repo_input_size, hidden_sizes_reponame_pipeline[0])
        self.repo_bn1 = nn.BatchNorm1d(hidden_sizes_reponame_pipeline[0])
        self.repo_relu1 = nn.ReLU()
        self.repo_dropout1 = nn.Dropout(dropout_rate)
        
        self.repo_fc2 = nn.Linear(hidden_sizes_reponame_pipeline[0], hidden_sizes_reponame_pipeline[1])
        self.repo_bn2 = nn.BatchNorm1d(hidden_sizes_reponame_pipeline[1])
        self.repo_relu2 = nn.ReLU()
        self.repo_dropout2 = nn.Dropout(dropout_rate)

        
        # Combining the outputs of both pipelines
        combined_input_size = hidden_sizes_embedding_pipeline[2] + hidden_sizes_reponame_pipeline[1]   # Since we're combining two pipelines
        self.combined_fc1 = nn.Linear(combined_input_size, hidden_sizes_merged_pipeline[0])
        self.combined_bn1 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[0])
        self.combined_relu1 = nn.ReLU()
        self.combined_dropout1 = nn.Dropout(dropout_rate)

        self.combined_fc2  = nn.Linear(hidden_sizes_merged_pipeline[0], hidden_sizes_merged_pipeline[1])
        self.combined_bn2 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[1])
        self.combined_relu2  = nn.ReLU()
        self.combined_dropout2 = nn.Dropout(dropout_rate)
        
        self.combined_fc3  = nn.Linear(hidden_sizes_merged_pipeline[1], hidden_sizes_merged_pipeline[2])
        self.combined_bn3 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[2])
        self.combined_relu3  = nn.ReLU()
        self.combined_dropout3 = nn.Dropout(dropout_rate)
        
        self.combined_fc4  = nn.Linear(hidden_sizes_merged_pipeline[2], hidden_sizes_merged_pipeline[3])
        self.combined_bn4 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[3])
        self.combined_relu4  = nn.ReLU()
        self.combined_dropout4 = nn.Dropout(dropout_rate)

        # Output layer
        self.fc_out = nn.Linear(hidden_sizes_merged_pipeline[3], output_size)
        
    def forward(self, code_x, repo_x):
        # Process code embeddings
        code_x = self.code_fc1(code_x)
        code_x = self.code_bn1(code_x)
        code_x = self.code_relu1(code_x)
        code_x = self.code_dropout1(code_x)
        
        code_x = self.code_fc2(code_x)
        code_x = self.code_bn2(code_x)
        code_x = self.code_relu2(code_x)
        code_x = self.code_dropout2(code_x)
        
        code_x = self.code_fc3(code_x)
        code_x = self.code_bn3(code_x)
        code_x = self.code_relu3(code_x)
        code_x = self.code_dropout3(code_x)
        
        # Process repo names
        repo_x = self.repo_fc1(repo_x)
        repo_x = self.repo_bn1(repo_x)
        repo_x = self.repo_relu1(repo_x)
        repo_x = self.repo_dropout1(repo_x)
        
        repo_x = self.repo_fc2(repo_x)
        repo_x = self.repo_bn2(repo_x)
        repo_x = self.repo_relu2(repo_x)
        repo_x = self.repo_dropout2(repo_x)
        
        # Combine both outputs
        combined_x = torch.cat((code_x, repo_x), dim=1)
        
        # Process the combined output
        combined_x = self.combined_fc1(combined_x)
        combined_x = self.combined_bn1(combined_x)
        combined_x = self.combined_relu1(combined_x)
        combined_x = self.combined_dropout1(combined_x)

        combined_x = self.combined_fc2(combined_x)
        combined_x = self.combined_bn2(combined_x)
        combined_x = self.combined_relu2(combined_x)
        combined_x = self.combined_dropout2(combined_x)
        
        combined_x = self.combined_fc3(combined_x)
        combined_x = self.combined_bn3(combined_x)
        combined_x = self.combined_relu3(combined_x)
        combined_x = self.combined_dropout3(combined_x)
        
        combined_x = self.combined_fc4(combined_x)
        combined_x = self.combined_bn4(combined_x)
        combined_x = self.combined_relu4(combined_x)
        combined_x = self.combined_dropout4(combined_x)
        
        # Final output
        output = self.fc_out(combined_x)
        return output



# Custom Dataset Class to handle the dataframe
class CodeDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.id_list = df['index'].values 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
    
        index = self.id_list[idx]  

        tensor_3d = self.df.loc[self.df['index'] == index, 'embedding'].iloc[0]
        repo_name_tensor = self.df.loc[self.df['index'] == index, 'repo_name_tensor'].iloc[0]
        label_tensor = self.df.loc[self.df['index'] == index, 'label_tensor'].iloc[0]
        
        #print(f"{repo_name_tensor.shape} {tensor_3d.shape} {label_tensor.shape}")
        #print("-------------------------------")
        
        #Flattening
        #embedding = torch.flatten(tensor_3d)
        embedding = tensor_3d
        return embedding, label_tensor, repo_name_tensor, index

# Creating a DataLoader
def create_dataloader(df, batch_size=32, shuffle=True):
    dataset = CodeDataset(df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def test_with_data_loader(model, loader, scaler, criterion):
    
    running_loss = 0.0
    running_accuracy = 0.0
    predicted_y = []
    y = []
    
    for batch_X, batch_y, batch_X_repo, idx in loader:
        
        batch_X = scaler.transform(batch_X)
        batch_X = torch.tensor(batch_X,dtype=torch.float32)

        # Forward pass
        batch_y = batch_y.to(device)
        batch_X = batch_X.to(device)
        batch_X_repo = batch_X_repo.to(device)
        
        # index encoding
        batch_y_1D = torch.argmax(batch_y, dim=1)

        outputs = model(batch_X, batch_X_repo)
        # outputs = model(batch_X)
        loss = criterion(outputs, batch_y_1D)
        
        # Calculate accuracy
        accuracy = calculate_accuracy(outputs, batch_y_1D)
        
        # Update running loss and accuracy
        running_loss += loss.item()
        running_accuracy += accuracy

        predicted_y.append(outputs)
        y.append(batch_y)
        
    test_loss = running_loss / len(loader)
    test_accuracy = running_accuracy / len(loader)
    
    return test_loss, test_accuracy, torch.cat(predicted_y, dim=0), torch.cat(y, dim=0)


# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
    # labels_1d = torch.argmax(labels, dim=1) # turn into 1D tensor
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)  # Accuracy = number of correct predictions / total number of samples
    return accuracy

def instantiate_NN_model(df, model_directory):
    
    directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_directory)
    code_embedding_dim = 768
    output_size = 3
    dropout_rate = 0.05
    repo_input_size = 626
    
    hidden_sizes_embedding_pipeline = [1024, 2048, 4096]
    hidden_sizes_reponame_pipeline = [512, 1024]
    hidden_sizes_merged_pipeline = [2048, 1024, 512, 256]

    # Load model from file
    model = CodeDiffNN(code_embedding_dim, repo_input_size, hidden_sizes_embedding_pipeline, 
                       hidden_sizes_reponame_pipeline, hidden_sizes_merged_pipeline, output_size, 
                       dropout_rate).to(device)
    model.load_state_dict(torch.load(directory))

    return model