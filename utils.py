from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, BertModel  # BertModel eklendi

embedding_directory = 'embeddings/GCB_Pooler_embeddings_patch_before_largewithcodedata_15300limit3label'
model_directory = 'models/two_pipelines_StScaler_pooler_6layers_invertednn_100epoch_batch_normalization.pth'
# Cihaz tanımı daha global ve esnek
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --- BERT Modelleri ve Tokenizer'ları Yükleme ---
# Her model için ayrı tokenizer ve model tanımlayabiliriz, veya tek bir tokenizer kullanabiliriz
# Metin tokenizasyonu genellikle benzer olduğu için tek bir tokenizer yeterli olabilir.
# Ancak embedding çıkarma stratejileri (CLS vs. Pooler) farklı olduğu için modelleri ayrı tutalım.
tokenizer_code = None
model_graphcodebert = None
tokenizer_text = None
model_bert_text = None


def load_bert_models_and_tokenizers():
    global tokenizer_code, model_graphcodebert, tokenizer_text, model_bert_text
    if tokenizer_code is None:
        print("Loading GraphCodeBERT tokenizer and model...")
        tokenizer_code = AutoTokenizer.from_pretrained(
            "microsoft/graphcodebert-base")
        model_graphcodebert = AutoModel.from_pretrained(
            "microsoft/graphcodebert-base")
        model_graphcodebert.to(device)
        model_graphcodebert.eval()
        print("GraphCodeBERT loaded.")

    if tokenizer_text is None:  # Düz BERT için ayrı bir tokenizer ve model
        print("Loading general BERT tokenizer and model for text...")
        # Metin için daha genel bir BERT modeli (örneğin bert-base-uncased) kullanabiliriz
        # Eğer repo isimleri çok özel bir dilde veya formatta ise başka bir model seçilebilir.
        tokenizer_text = AutoTokenizer.from_pretrained("bert-base-uncased")
        model_bert_text = BertModel.from_pretrained(
            "bert-base-uncased")  # Düz BertModel kullanıyoruz
        model_bert_text.to(device)
        model_bert_text.eval()
        print("General BERT model loaded.")


# utils.py

# Global modelleri yüklemek için cache dictionary
_models_cache = {}

# Cihaz tanımı: GPU varsa 'cuda:0', yoksa 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_bert_embedding(text, model_type="text"):
    """
    Belirtilen model tipi için metnin BERT CLS token embedding'ini döndürür.
    `model_type` "text" veya "graphcodebert" olabilir.
    """
    if model_type == "text":
        model_name = "bert-base-uncased"
        print("Loading general BERT tokenizer and model for text...")
    elif model_type == "graphcodebert":
        model_name = "microsoft/graphcodebert-base"
        print("Loading GraphCodeBERT tokenizer and model...")
    else:
        raise ValueError("model_type 'text' veya 'graphcodebert' olmalı.")

    # Modelleri önbellekten al veya yükle
    if model_name not in _models_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)  # Modeli cihaza taşı
        model.eval()  # Değerlendirme moduna al
        _models_cache[model_name] = {'tokenizer': tokenizer, 'model': model}
        if model_type == "text":
            print("General BERT model loaded.")
        else:
            print("GraphCodeBERT loaded.")
    else:
        print(f"Using cached model for {model_name}.")

    tokenizer = _models_cache[model_name]['tokenizer']
    model = _models_cache[model_name]['model']

    print(f"Tokenizing text for {model_name}...")
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True, max_length=512)

    # Giriş tensörlerini modele göndermeden önce doğru cihaza taşı
    # inputs = {k: v.to(device) for k, v in inputs.items()} # Bu satır gerekli değil çünkü PyTorch'a göre işlem yapıyoruz
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    print(f"Input IDs device: {input_ids.device}")
    print(f"Attention Mask device: {attention_mask.device}")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    print(f"Model outputs device: {outputs.last_hidden_state.device}")

    # CLS token'ın embedding'ini al
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # Embedding'in cihazını kontrol edelim
    print(f"CLS embedding device BEFORE return: {cls_embedding.device}")

    return cls_embedding


def prepare_dataframe(file_path, should_balance=False, should_merge_csv=False):
    # Bu fonksiyon artık manuel girişlerden DataFrame oluşturulduğu için çağrılmayacak.
    # Ancak yine de kodda durmasında bir sakınca yok.
    print(f"Loading dataframe from {file_path}")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=[
                          'labels', 'full_repo_name', 'patch', 'file_path', 'index', 'content_before'])
        print(
            f"Warning: {file_path} not found. Returning empty dataframe for now.")
    return df


def onehot_embed_df(df):
    # Bu fonksiyon artık repo adı embedding'i için kullanılmayacak.
    # Sadece etiketleri one-hot encode etmeye devam edebiliriz, ama manuel girişte etiket yok.
    # main.py'de dummy etiket oluşturduğumuz için bu fonksiyonun manuel tahmin akışında bir rolü kalmadı.
    print("Warning: onehot_embed_df is being called. This function is mostly deprecated for manual input flow.")

    # Eğer hala 'labels' sütunu varsa ve one-hot encode edilmesi gerekiyorsa:
    if 'labels' in df.columns:
        df_encoded_labels = pd.get_dummies(df, columns=['labels'])
        y = df_encoded_labels.drop(columns=[col for col in [
                                   'patch', 'full_repo_name', 'index', 'file_path', 'content_before'] if col in df_encoded_labels.columns])
        y = y.astype(int)
        df['label_tensor'] = [torch.tensor(
            row, dtype=torch.float32) for row in y.values]
    else:
        # Eğer 'labels' sütunu yoksa (manuel tahmin durumunda), dummy label_tensor ekleyelim
        df['label_tensor'] = [torch.tensor(
            [0, 0, 0], dtype=torch.float32)] * len(df)  # 3 sınıf için dummy

    # 'repo_name_tensor' artık BERT ile oluşturulacağı için buradan kaldırıldı.
    # 'embedding' (code embedding) de doğrudan main.py'de eklendiği için buradan kaldırıldı.

    # Eğer bu sütunlar varsa düşür
    return df.drop(columns=[col for col in ['labels', 'patch'] if col in df.columns])


def load_embeddings_from_directory_for_df(directory_path, df):
    # Bu fonksiyon manuel tahmin akışında kullanılmayacak.
    print(f"Warning: load_embeddings_from_directory_for_df is being called. This function is deprecated for manual input flow.")
    # Sadece 'embedding' sütunu yoksa, dummy embedding ekleyelim
    if 'embedding' not in df.columns:
        df['embedding'] = [torch.zeros(768).to(device)] * len(df)
    return df


# Define the ANN model (Bu kısım aynı kalabilir)
class CodeDiffNN(nn.Module):
    def __init__(self, code_input_size, repo_input_size, hidden_sizes_embedding_pipeline, hidden_sizes_reponame_pipeline, hidden_sizes_merged_pipeline, output_size, dropout_rate):
        super(CodeDiffNN, self).__init__()

        # Pipeline for code embeddings
        self.code_fc1 = nn.Linear(
            code_input_size, hidden_sizes_embedding_pipeline[0])
        self.code_bn1 = nn.BatchNorm1d(hidden_sizes_embedding_pipeline[0])
        self.code_relu1 = nn.ReLU()
        self.code_dropout1 = nn.Dropout(dropout_rate)

        self.code_fc2 = nn.Linear(
            hidden_sizes_embedding_pipeline[0], hidden_sizes_embedding_pipeline[1])
        self.code_bn2 = nn.BatchNorm1d(hidden_sizes_embedding_pipeline[1])
        self.code_relu2 = nn.ReLU()
        self.code_dropout2 = nn.Dropout(dropout_rate)

        self.code_fc3 = nn.Linear(
            hidden_sizes_embedding_pipeline[1], hidden_sizes_embedding_pipeline[2])
        self.code_bn3 = nn.BatchNorm1d(hidden_sizes_embedding_pipeline[2])
        self.code_relu3 = nn.ReLU()
        self.code_dropout3 = nn.Dropout(dropout_rate)

        # Pipeline for repo names
        self.repo_fc1 = nn.Linear(
            repo_input_size, hidden_sizes_reponame_pipeline[0])
        self.repo_bn1 = nn.BatchNorm1d(hidden_sizes_reponame_pipeline[0])
        self.repo_relu1 = nn.ReLU()
        self.repo_dropout1 = nn.Dropout(dropout_rate)

        self.repo_fc2 = nn.Linear(
            hidden_sizes_reponame_pipeline[0], hidden_sizes_reponame_pipeline[1])
        self.repo_bn2 = nn.BatchNorm1d(hidden_sizes_reponame_pipeline[1])
        self.repo_relu2 = nn.ReLU()
        self.repo_dropout2 = nn.Dropout(dropout_rate)

        # Combining the outputs of both pipelines
        combined_input_size = hidden_sizes_embedding_pipeline[2] + \
            hidden_sizes_reponame_pipeline[1]
        self.combined_fc1 = nn.Linear(
            combined_input_size, hidden_sizes_merged_pipeline[0])
        self.combined_bn1 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[0])
        self.combined_relu1 = nn.ReLU()
        self.combined_dropout1 = nn.Dropout(dropout_rate)

        self.combined_fc2 = nn.Linear(
            hidden_sizes_merged_pipeline[0], hidden_sizes_merged_pipeline[1])
        self.combined_bn2 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[1])
        self.combined_relu2 = nn.ReLU()
        self.combined_dropout2 = nn.Dropout(dropout_rate)

        self.combined_fc3 = nn.Linear(
            hidden_sizes_merged_pipeline[1], hidden_sizes_merged_pipeline[2])
        self.combined_bn3 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[2])
        self.combined_relu3 = nn.ReLU()
        self.combined_dropout3 = nn.Dropout(dropout_rate)

        self.combined_fc4 = nn.Linear(
            hidden_sizes_merged_pipeline[2], hidden_sizes_merged_pipeline[3])
        self.combined_bn4 = nn.BatchNorm1d(hidden_sizes_merged_pipeline[3])
        self.combined_relu4 = nn.ReLU()
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
        if 'index' not in self.df.columns:
            self.df['index'] = range(len(self.df))
        self.id_list = df['index'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        index = self.id_list[idx]

        embedding = self.df.loc[self.df['index'] == index, 'embedding'].iloc[0]
        repo_name_tensor = self.df.loc[self.df['index']
                                       == index, 'repo_name_tensor'].iloc[0]
        label_tensor = self.df.loc[self.df['index']
                                   == index, 'label_tensor'].iloc[0]

        # Ensure embedding and repo_name_tensor are on the correct device for the model.
        # This will move them to GPU if available before returning.
        embedding = embedding.to(device)
        repo_name_tensor = repo_name_tensor.to(device)
        # Label tensoru da aynı cihaza taşı
        label_tensor = label_tensor.to(device)

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
        # Scale the code embeddings. Ensure batch_X is on CPU before scaling, then move back to device.
        # scaler.transform expects a NumPy array or something that can be converted to one.
        # So, we first move batch_X to CPU, convert to NumPy, scale, and then convert back to tensor and move to device.
        batch_X_cpu = batch_X.cpu().numpy()  # GPU'dan CPU'ya taşı ve NumPy'ye çevir
        batch_X_scaled_np = scaler.transform(batch_X_cpu)
        batch_X = torch.tensor(batch_X_scaled_np, dtype=torch.float32).to(
            device)  # Tensöre çevir ve tekrar GPU'ya taşı

        # Forward pass
        # batch_y, batch_X, batch_X_repo are already on the correct device from CodeDataset's __getitem__
        # batch_y = batch_y.to(device) # No longer needed here
        # batch_X = batch_X.to(device) # No longer needed here
        # batch_X_repo = batch_X_repo.to(device) # No longer needed here

        # index encoding
        # batch_y'nin zaten device üzerinde olduğunu varsayıyoruz
        batch_y_1D = torch.argmax(batch_y, dim=1)

        outputs = model(batch_X, batch_X_repo)
        loss = criterion(outputs, batch_y_1D)

        # Calculate accuracy
        accuracy = calculate_accuracy(outputs, batch_y_1D)

        # Update running loss and accuracy
        running_loss += loss.item()
        running_accuracy += accuracy

        # Append outputs and y to lists (ensure they are on CPU if you'll concatenate later for NumPy operations)
        predicted_y.append(outputs.cpu())
        y.append(batch_y.cpu())

    test_loss = running_loss / len(loader)
    test_accuracy = running_accuracy / len(loader)

    return test_loss, test_accuracy, torch.cat(predicted_y, dim=0), torch.cat(y, dim=0)


# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


def instantiate_NN_model(df, model_directory):

    directory = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), model_directory)
    code_embedding_dim = 768
    output_size = 3
    dropout_rate = 0.05
    repo_input_size = 768

    hidden_sizes_embedding_pipeline = [1024, 2048, 4096]
    hidden_sizes_reponame_pipeline = [512, 1024]
    hidden_sizes_merged_pipeline = [2048, 1024, 512, 256]

    # Load model from file
    model = CodeDiffNN(code_embedding_dim, repo_input_size, hidden_sizes_embedding_pipeline,
                       hidden_sizes_reponame_pipeline, hidden_sizes_merged_pipeline, output_size,
                       dropout_rate).to(device)
    model.load_state_dict(torch.load(directory))

    return model
