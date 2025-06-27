
import json
import random
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import torch
import torch.nn as nn
import os

# Download NLTK tokenizer
nltk.download('punkt', force=True)  # Force fresh download of the correct data

# Setup stemmer and file paths
stemmer = PorterStemmer()
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
file_path = os.path.join(current_dir, "intents.json")

# Load intents
with open(file_path, "r") as file:
    data = json.load(file)

# Tokenization and stemming
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    return np.array([1 if w in sentence_words else 0 for w in all_words], dtype=np.float32)

# Prepare training data
all_words = []
tags = []
xy = []

for intent in data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!', ',']
all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bow = bag_of_words(pattern_sentence, all_words)
    X_train.append(bow)
    y_train.append(tags.index(tag))

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Define the model
class ChatNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)

# Initialize and train model
model = ChatNet(input_size=len(all_words), hidden_size=500, output_size=len(tags))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10000
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training complete.")

# Predict function
def predict_class(sentence):
    sentence_bow = bag_of_words(tokenize(sentence), all_words)
    input_tensor = torch.tensor(sentence_bow, dtype=torch.float32).unsqueeze(0)
    output = model(input_tensor)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    return tag

# Search simulation
def google_search(query):
    return "Search results here (connect API)"

# Chat function
def chat():
    print("Chatbot ready! Type 'quit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        tag = predict_class(user_input)
        found = False
        for intent in data['intents']:
            if tag == intent["tag"]:
                found = True
                if tag == "search":
                    print("Bot:", google_search(user_input))
                else:
                    print("Bot:", random.choice(intent["responses"]))
                break
        if not found:
            print("Bot: Sorry, I don't understand that.")

chat()
