import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score

# Load the datasets
restaurants_df = pd.read_csv('/content/Restaurants_Train_v2.csv')
laptops_df = pd.read_csv('/content/Laptop_Train_v2.csv')

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Map string labels to numerical values
label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
restaurants_df['polarity'] = restaurants_df['polarity'].map(label_mapping)
laptops_df['polarity'] = laptops_df['polarity'].map(label_mapping)

# Function to validate the labels
def validate_labels(df):
    invalid_labels = df[~df['polarity'].isin([0, 1, 2])]
    if not invalid_labels.empty:
        print("Invalid labels found:")
        print(invalid_labels)
    else:
        print("All labels are valid.")

# Validate the labels in both datasets
validate_labels(restaurants_df)
validate_labels(laptops_df)

# Handle NaN or invalid labels
restaurants_df['polarity'].fillna(1, inplace=True)
restaurants_df = restaurants_df[restaurants_df['polarity'].isin([0, 1, 2])]

laptops_df['polarity'].fillna(1, inplace=True)
laptops_df = laptops_df[laptops_df['polarity'].isin([0, 1, 2])]

# Dataset class
class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sentence = self.df.iloc[idx]['Sentence']
        aspect = self.df.iloc[idx]['Aspect Term']
        label = self.df.iloc[idx]['polarity']

        # Debugging print statement for labels
        print(f"Processing sentence {idx}: Label = {label}")

        inputs = self.tokenizer.encode_plus(
            sentence,
            aspect,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Hybrid Model combining BERT and a custom classifier
class HybridModel(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(HybridModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to train and evaluate the model
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, device, epochs=5, patience=2):
    best_accuracy = 0
    best_epoch = 0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_accuracy = evaluate(model, val_loader, device)
        print(f'Epoch {epoch + 1} - Validation Accuracy: {val_accuracy * 100:.2f}%')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

    print(f'Best Validation Accuracy: {best_accuracy * 100:.2f}% at epoch {best_epoch + 1}')
    return best_accuracy

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds)

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# Hyperparameters
max_len = 128
batch_size = 16
learning_rate = 2e-5
epochs = 20
patience = 15

# Prepare the dataset
restaurant_dataset = ABSADataset(restaurants_df, tokenizer, max_len)
train_size = int(0.8 * len(restaurant_dataset))
val_size = len(restaurant_dataset) - train_size
train_dataset, val_dataset = random_split(restaurant_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = BertModel.from_pretrained('bert-base-uncased')
model = HybridModel(bert_model, num_labels=3)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Train and evaluate the model
best_accuracy = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, device, epochs, patience)
