from datasets import load_dataset
from torch import nn
import torch
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, DataLoader
import wandb


# This project is meant to generate next token given a sequence of words

# Set up the weights and biases code
run = wandb.init(
    name="Add extra metrics",
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="maxpendse-projects",
    # Set the wandb project where this run will be logged.
    project="Max-LLM-Project",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.001,
        "epochs": 1,
        "dataset": "WikiLab Dataset",
        "batch_size": 100,
    },
)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
with open('pretrain_transformer/unique_chars_wikitext.txt', 'r') as f:
    chars = list(f.read())

# Hyperparameters
block_size = 100
embedding_size = 768
num_heads = 3
head_dim = 1000
vocab_size = len(chars)
epochs = run.config.epochs

# Define the Model Architecture that we're using

class AttentionHead(nn.Module):
    def __init__(self, embed_size, head_dim):
        super().__init__()
        self.Q = nn.Linear(embed_size, head_dim, bias=False)
        self.K = nn.Linear(embed_size, head_dim, bias=False)
        self.V = nn.Linear(embed_size, head_dim, bias=False)
        self.scale = head_dim ** -0.5

    def forward(self, x):
        q = self.Q(x)  # (B, T, head_dim)
        k = self.K(x)  # (B, T, head_dim)
        v = self.V(x)  # (B, T, head_dim)

        attn = q @ k.transpose(-2, -1) * self.scale  # (B, T, T)
        attn = softmax(attn, dim=-1)
        return attn @ v  # (B, T, head_dim)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, head_dim):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(embed_size, head_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_dim, embed_size)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

# Initialize the model
class TextTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)

        self.attention1 = MultiHeadAttention(embedding_size, num_heads, embedding_size//num_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ff1 = nn.Sequential(nn.Linear(embedding_size, 4 * embedding_size), nn.GELU(), nn.Linear(4 * embedding_size, embedding_size))
        self.norm1b = nn.LayerNorm(embedding_size)

        self.attention2 = MultiHeadAttention(embedding_size, num_heads, embedding_size//num_heads)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.ff2 = nn.Sequential(nn.Linear(embedding_size, 4 * embedding_size), nn.GELU(), nn.Linear(4 * embedding_size, embedding_size))
        self.norm2b = nn.LayerNorm(embedding_size)

        self.attention3 = MultiHeadAttention(embedding_size, num_heads, embedding_size//num_heads)
        self.norm3 = nn.LayerNorm(embedding_size)
        self.ff3 = nn.Sequential(nn.Linear(embedding_size, 4 * embedding_size), nn.GELU(), nn.Linear(4 * embedding_size, embedding_size))
        self.norm3b = nn.LayerNorm(embedding_size)

        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, seq):
        x = self.embed(seq)
        x = self.norm1(x + self.attention1(x))
        x = self.norm1b(x + self.ff1(x))
        x = self.norm2(x + self.attention2(x))
        x = self.norm2b(x + self.ff2(x))
        x = self.norm3(x + self.attention3(x))
        x = self.norm3b(x + self.ff3(x))

        # return the argmax w.r.t. the vocab
        return self.linear(x)
    

# Data preprocessing
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
text_train = ds['train']
text_test = ds['test']

def dataset_to_sequences(dataset, block_size):
    text = "\n".join(item["text"] for item in dataset if item["text"])
    num_blocks = len(text) // block_size
    return [text[i * block_size:(i + 1) * block_size] for i in range(num_blocks)]

blocked_train_data = dataset_to_sequences(text_train, block_size)
blocked_test_data = dataset_to_sequences(text_test, block_size)

character_to_number_encoding_map = {ch: i for i, ch in enumerate(chars)}
encode = lambda s: [character_to_number_encoding_map[c] for c in s]
blocked_train_data_encoded = [encode(sequence) for sequence in blocked_train_data]
blocked_test_data_encoded = [encode(sequence) for sequence in blocked_test_data]



# Initialize the model on the device
model = TextTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=run.config.learning_rate)
loss_fn = nn.CrossEntropyLoss()
epoch_loss = 0

model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for batch_idx in range(0, len(blocked_train_data_encoded), run.config.batch_size):
        batch = blocked_train_data_encoded[batch_idx: batch_idx + run.config.batch_size]

        # We need to take the first (batch_size - 1) elements because the last element has no y pair
        # We are training to find the optimal y character for the previous x character -- why we shift by 1. 
        # We need to convert these y values to one hot vectors
        batch = torch.tensor(batch, dtype=torch.long).to(device)

        batch_x = batch[:, :-1]
        batch_y = torch.nn.functional.one_hot(batch[:, 1:], num_classes=vocab_size).float()
        batch_y = batch_y.permute(0,2,1)

        optimizer.zero_grad()

        model_outputs = model(batch_x)
        model_outputs = model_outputs.permute(0,2,1)

        loss = loss_fn(model_outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss * len(batch_x)

        print(f'The train loss for batch: {batch_idx} is {loss}')

        if batch_idx > 12000:
            break


# --- Inference ---
decode = lambda tokens: "".join(chars[i] for i in tokens)

def generate(prompt: str, num_tokens: int = 50) -> str:
    model.eval()
    with torch.no_grad():
        token_ids = encode(prompt)
        for _ in range(num_tokens):
            # Truncate to block_size - 1 (model was trained on sequences of length block_size - 1)
            context = token_ids[-(block_size - 1):]
            x = torch.tensor([context], dtype=torch.long).to(device)
            logits = model(x)           # (1, T, vocab_size)
            next_token = logits[0, -1].argmax().item()
            token_ids.append(next_token)
    return decode(token_ids)

# Interactive prompt loop
print("\n--- Model Inference ---")
print("Type a prompt and press Enter. Type 'quit' to exit.\n")
while True:
    prompt = input("Prompt: ")
    if prompt.lower() == "quit":
        break
    try:
        output = generate(prompt, num_tokens=100)
        print(f"Output: {output}\n")
    except KeyError as e:
        print(f"Character {e} not in vocabulary. Use only characters the model was trained on.\n")
