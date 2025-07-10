import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn as nn
import torch
import pandas as pd
print("DNN.PY YÜKLENDİ - VERSİYON 1.0")

embedding_directory = 'embeddings/GCB_Pooler_embeddings_patch_before_largewithcodedata_15300limit3label'
model_directory = 'models/two_pipelines_StScaler_pooler_6layers_invertednn_100epoch_batch_normalization.pth'
# Cihaz tanımı: GPU varsa 'cuda:0', yoksa 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Bu fonksiyon artık utils.py'den çağrılmayacak veya main.py'de manuel olarak veri oluşturulduğu için
# doğrudan kullanılmayacak. Ancak mevcut olduğu için içeriği değişmeden kalabilir.


def onehot_embed_df(df):
    # One-hot encoding ile labels sütununu dönüştürme
    df_encoded_labels = pd.get_dummies(df, columns=['labels'])
    df_encoded_repo = pd.get_dummies(df, columns=['full_repo_name'])

    # Özellikler ve etiketler
    X_repo = df_encoded_repo.drop(
        columns=['labels', 'patch', 'index', 'file_path'])
    X_repo = X_repo.astype(int)

    y = df_encoded_labels.drop(
        columns=['patch', 'full_repo_name', 'index', 'file_path'])
    y = y.astype(int)

    df['label_tensor'] = [torch.tensor(
        row, dtype=torch.float32) for row in y.values]
    df['repo_name_tensor'] = [torch.tensor(
        row, dtype=torch.float32) for row in X_repo.values]

    return df.drop(columns=['labels', 'patch'])


# Define the ANN model# dnn.py

# Cihaz tanımı: GPU varsa 'cuda:0', yoksa 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Define the ANN model - GÜNCELLENMİŞ VERSİYON


# dnn.py

# Cihaz tanımı: GPU varsa 'cuda:0', yoksa 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Define the ANN model - GÜNCELLENMİŞ VERSİYON (instantiate_NN_model ile EŞLEŞTİRİLDİ)


class CodeDiffNN(nn.Module):
    def __init__(self, code_input_size=768, repo_input_size=768,
                 hidden_sizes_embedding_pipeline=[
                     1024, 2048, 4096],  # Senin eğitimdeki değerler
                 hidden_sizes_reponame_pipeline=[
                     512, 1024],  # Senin eğitimdeki değerler
                 # Senin eğitimdeki değerler
                 hidden_sizes_merged_pipeline=[2048, 1024, 512, 256],
                 output_size=3, dropout_rate=0.05):  # Senin eğitimdeki değerler
        super(CodeDiffNN, self).__init__()

        # Pipeline for code embeddings
        self.code_fc1 = nn.Linear(
            # 1536 -> 1024
            code_input_size, hidden_sizes_embedding_pipeline[0])
        self.code_bn1 = nn.BatchNorm1d(
            hidden_sizes_embedding_pipeline[0])  # 1024
        self.code_relu1 = nn.ReLU()
        self.code_dropout1 = nn.Dropout(dropout_rate)

        self.code_fc2 = nn.Linear(
            # 1024 -> 2048
            hidden_sizes_embedding_pipeline[0], hidden_sizes_embedding_pipeline[1])
        self.code_bn2 = nn.BatchNorm1d(
            hidden_sizes_embedding_pipeline[1])  # 2048
        self.code_relu2 = nn.ReLU()
        self.code_dropout2 = nn.Dropout(dropout_rate)

        self.code_fc3 = nn.Linear(
            # 2048 -> 4096
            hidden_sizes_embedding_pipeline[1], hidden_sizes_embedding_pipeline[2])
        self.code_bn3 = nn.BatchNorm1d(
            hidden_sizes_embedding_pipeline[2])  # 4096
        self.code_relu3 = nn.ReLU()
        self.code_dropout3 = nn.Dropout(dropout_rate)

        # Pipeline for repo names
        self.repo_fc1 = nn.Linear(
            repo_input_size, hidden_sizes_reponame_pipeline[0])  # 768 -> 512
        self.repo_bn1 = nn.BatchNorm1d(
            hidden_sizes_reponame_pipeline[0])  # 512
        self.repo_relu1 = nn.ReLU()
        self.repo_dropout1 = nn.Dropout(dropout_rate)

        self.repo_fc2 = nn.Linear(
            # 512 -> 1024
            hidden_sizes_reponame_pipeline[0], hidden_sizes_reponame_pipeline[1])
        self.repo_bn2 = nn.BatchNorm1d(
            hidden_sizes_reponame_pipeline[1])  # 1024
        self.repo_relu2 = nn.ReLU()
        self.repo_dropout2 = nn.Dropout(dropout_rate)

        # Combining the outputs of both pipelines
        # code_fc3 çıktısı: hidden_sizes_embedding_pipeline[2] = 4096
        # repo_fc2 çıktısı: hidden_sizes_reponame_pipeline[1] = 1024
        # 4096 + 1024 = 5120
        combined_input_size = hidden_sizes_embedding_pipeline[2] + \
            hidden_sizes_reponame_pipeline[1]
        self.combined_fc1 = nn.Linear(
            # 5120 -> 2048
            combined_input_size, hidden_sizes_merged_pipeline[0])
        self.combined_bn1 = nn.BatchNorm1d(
            hidden_sizes_merged_pipeline[0])  # 2048
        self.combined_relu1 = nn.ReLU()
        self.combined_dropout1 = nn.Dropout(dropout_rate)

        self.combined_fc2 = nn.Linear(
            # 2048 -> 1024
            hidden_sizes_merged_pipeline[0], hidden_sizes_merged_pipeline[1])
        self.combined_bn2 = nn.BatchNorm1d(
            hidden_sizes_merged_pipeline[1])  # 1024
        self.combined_relu2 = nn.ReLU()
        self.combined_dropout2 = nn.Dropout(dropout_rate)

        self.combined_fc3 = nn.Linear(
            # 1024 -> 512
            hidden_sizes_merged_pipeline[1], hidden_sizes_merged_pipeline[2])
        self.combined_bn3 = nn.BatchNorm1d(
            hidden_sizes_merged_pipeline[2])  # 512
        self.combined_relu3 = nn.ReLU()
        self.combined_dropout3 = nn.Dropout(dropout_rate)

        self.combined_fc4 = nn.Linear(
            # 512 -> 256
            hidden_sizes_merged_pipeline[2], hidden_sizes_merged_pipeline[3])
        self.combined_bn4 = nn.BatchNorm1d(
            hidden_sizes_merged_pipeline[3])  # 256
        self.combined_relu4 = nn.ReLU()
        self.combined_dropout4 = nn.Dropout(dropout_rate)

        # Output layer
        self.fc_out = nn.Linear(
            hidden_sizes_merged_pipeline[3], output_size)  # 256 -> 3

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

# ... (geri kalan dnn.py kodları - CodeDataset, create_dataloader, calculate_accuracy, test_with_data_loader) ...

# Custom Dataset Class to handle the dataframe


class CodeDataset(Dataset):
    def __init__(self, df):
        self.df = df
        if 'index' not in self.df.columns:
            self.df['index'] = range(len(self.df))
        self.id_list = df['index'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.id_list[idx]

        # DataFrame'den tensörleri al
        embedding = self.df.loc[self.df['index'] == index, 'embedding'].iloc[0]
        repo_name_tensor = self.df.loc[self.df['index']
                                       == index, 'repo_name_tensor'].iloc[0]
        label_tensor = self.df.loc[self.df['index']
                                   == index, 'label_tensor'].iloc[0]

        print(
            f"In CodeDataset.__getitem__: Original embedding device: {embedding.device}")
        print(
            f"In CodeDataset.__getitem__: Original repo_name_tensor device: {repo_name_tensor.device}")
        print(
            f"In CodeDataset.__getitem__: Original label_tensor device: {label_tensor.device}")

        # Tensörleri doğru cihaza taşı (GPU veya CPU)
        # embedding ve repo_name_tensor zaten main.py'de .to(device) ile oluşturulduğu için
        # burada tekrar .to(device) çağırmak fazladan işlem yükü getirebilir.
        # Ancak emin olmak için bırakılabilir, PyTorch tensör zaten doğru cihazdaysa bir şey yapmaz.
        embedding = embedding.to(device)
        repo_name_tensor = repo_name_tensor.to(device)
        label_tensor = label_tensor.to(device)

        print(
            f"In CodeDataset.__getitem__: After .to(device) - embedding device: {embedding.device}")
        print(
            f"In CodeDataset.__getitem__: After .to(device) - repo_name_tensor device: {repo_name_tensor.device}")
        print(
            f"In CodeDataset.__getitem__: After .to(device) - label_tensor device: {label_tensor.device}")

        return embedding, label_tensor, repo_name_tensor, index
# Creating a DataLoader


def create_dataloader(df, batch_size=32, shuffle=True):
    dataset = CodeDataset(df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# İki scaler alacak şekilde güncellendi
def test_with_data_loader(model, loader, scaler_code, scaler_repo, criterion):
    running_loss = 0.0
    running_accuracy = 0.0
    predicted_y = []
    y = []

    model.eval()  # Modeli değerlendirme moduna ayarla

    with torch.no_grad():  # Gradyan hesaplamalarını devre dışı bırak
        for batch_X, batch_y, batch_X_repo, idx in loader:
            # Verileri doğru cihaza taşı
            # batch_X, batch_y, batch_X_repo already moved to device in CodeDataset.__getitem__

            # print("Inside test_with_data_loader loop:")
            # print(f"  batch_X device: {batch_X.device}, shape: {batch_X.shape}")
            # print(f"  batch_y device: {batch_y.device}, shape: {batch_y.shape}")
            # print(f"  batch_X_repo device: {batch_X_repo.device}, shape: {batch_X_repo.shape}")

            # Code embeddings için scaler
            # batch_X boyutu [batch_size, 2, 768] ise [batch_size, 1536] olarak yeniden şekillendir
            if batch_X.ndim == 3 and batch_X.shape[1] == 2:
                batch_X_reshaped = batch_X.view(
                    batch_X.shape[0], -1)  # [batch_size, 1536]
            else:
                batch_X_reshaped = batch_X

            batch_X_cpu = batch_X_reshaped.cpu().numpy()
            batch_X_scaled_np = scaler_code.transform(
                batch_X_cpu)  # scaler_code kullanılıyor
            batch_X = torch.tensor(
                batch_X_scaled_np, dtype=torch.float32).to(device)
            # print(f"  batch_X after scaling: {batch_X.device}, shape: {batch_X.shape}")

            # Repo name embeddings için scaler
            # batch_X_repo boyutu [batch_size, 1, 768] ise [batch_size, 768] olarak yeniden şekillendir
            if batch_X_repo.ndim == 3 and batch_X_repo.shape[1] == 1:
                batch_X_repo_reshaped = batch_X_repo.view(
                    batch_X_repo.shape[0], -1)  # [batch_size, 768]
            else:
                batch_X_repo_reshaped = batch_X_repo

            batch_X_repo_cpu = batch_X_repo_reshaped.cpu().numpy()
            batch_X_repo_scaled_np = scaler_repo.transform(
                batch_X_repo_cpu)  # scaler_repo kullanılıyor
            batch_X_repo_final = torch.tensor(
                batch_X_repo_scaled_np, dtype=torch.float32).to(device)
            # print(f"  batch_X_repo after scaling: {batch_X_repo_final.device}, shape: {batch_X_repo_final.shape}")

            # One-hot encoded etiketi tek bir sınıf indeksine dönüştür
            batch_y_1D = torch.argmax(batch_y, dim=1)
            # print(f"  batch_y_1D device: {batch_y_1D.device}, shape: {batch_y_1D.shape}")

            # Modeli çağır, şimdi iki ölçeklenmiş input ile
            outputs = model(batch_X, batch_X_repo_final)
            # print(f"  Model outputs device: {outputs.device}, shape: {outputs.shape}")

            # Loss hesaplaması (dummy criterion kullanılıyorsa önemsiz)
            loss = criterion(outputs, batch_y_1D)

            accuracy = calculate_accuracy(outputs, batch_y_1D)

            running_loss += loss.item()
            running_accuracy += accuracy

            predicted_y.append(outputs.cpu())
            y.append(batch_y.cpu())

    test_loss = running_loss / len(loader)
    test_accuracy = running_accuracy / len(loader)

    final_predicted_y = torch.cat(predicted_y, dim=0)
    final_y = torch.cat(y, dim=0)

    return test_loss, test_accuracy, final_predicted_y, final_y

# Function to calculate accuracy


def calculate_accuracy(outputs, labels):
    # En yüksek olasılığa sahip sınıfın indeksini al
    _, predicted = torch.max(outputs, 1)
    # Doğru tahmin edilenlerin sayısını bul
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)  # Doğruluk yüzdesini hesapla
    return accuracy

# main.py
# ...


def instantiate_NN_model(model_directory):
    directory = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), model_directory)
    code_embedding_dim = 768
    output_size = 3
    dropout_rate = 0.05
    repo_input_size = 768

    # Bunların dnn.py'deki CodeDiffNN'in __init__ varsayılanlarıyla aynı olduğundan emin ol!
    hidden_sizes_embedding_pipeline = [1024, 2048, 4096]
    hidden_sizes_reponame_pipeline = [512, 1024]
    hidden_sizes_merged_pipeline = [2048, 1024, 512, 256]

    model = CodeDiffNN(code_embedding_dim, repo_input_size, hidden_sizes_embedding_pipeline,
                       hidden_sizes_reponame_pipeline, hidden_sizes_merged_pipeline, output_size,
                       dropout_rate).to(device)
    model.load_state_dict(torch.load(directory, map_location=device))
    return model

# ...
