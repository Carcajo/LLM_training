import pandas as pd
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification


def load_data(filepath):
    df = pd.read_csv(filepath)
    texts = df['text'].values
    labels = df['label'].values
    return texts, labels


class ReviewDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        if idx >= len(self.encodings):
            raise IndexError
        item = {...}  # код как раньше
        return item

    def __len__(self):
        return len(self.labels)



train_texts = [
  "This movie had excellent acting and a great story",
  "The cinematography was pretty good but the plot was boring",
]

train_labels = [1, 0, ...]

val_texts = [
  "The director ruined this otherwise interesting film",
  "I almost fell asleep watching this movie, do not recommend",
  # etc
]

val_labels = [0, 0, ...]



tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<PAD>')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)


train_dataset = ReviewDataset(train_encodings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = ReviewDataset(val_encodings, val_labels)
val_loader = DataLoader(val_dataset, batch_size=16)


model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(3):

    for batch in train_loader:
        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()


review_text = "This was the worst movie ever"
encoded = tokenizer.encode(review_text, return_tensors='pt')
output = model(encoded)
print(output.logits.argmax().item())