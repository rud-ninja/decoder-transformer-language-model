import sys
import time


import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer, Tokenizer

tokenizer = Tokenizer.from_file(r"C:\\Users\\hp\\Downloads\\secondapp\\latest check\\bpe.tokenizer_1.json")

vocab_size = 300
block_size = 256
batch_size = 64
n_embd = 512
ma_head = 8
n_blocks = 4
learning_rate = 3e-3
max_iters = 6000
eval_interval = 500
device = "cuda" if  torch.cuda.is_available() else "cpu"
eval_iters = 100
dropout = 0.2

# torch.manual_seed(9)


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out



class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out



class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=ma_head) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets==None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, tokenizer, max_new_tokens):
        # def update_text_and_print(new_list):
        #     if type(new_list) == 'list':
        #         for ind in new_list:
        #             subword = tokenizer.decode(ind)
        #             sys.stdout.write(subword)
        #             sys.stdout.flush()
        #             time.sleep(0.05)
        #     else:
        #         subword = tokenizer.decode(new_list)
        #         sys.stdout.write(subword)
        #         sys.stdout.flush()
        #         time.sleep(0.05)


        # update_text_and_print(idx[0].tolist())
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield tokenizer.decode(idx_next[0].tolist())
            # update_text_and_print(idx_next[0].tolist())

        return None


model = BigramLanguageModel(vocab_size)
model.load_state_dict(torch.load(r"C:\\Users\\hp\\Downloads\\secondapp\\latest check\\llmodel_1.pt", map_location=torch.device(device)))
model.eval()

my_input = "Who should be"
print(f"Your input context is: {my_input}", end="\n\n")
context = torch.tensor(tokenizer.encode(my_input).ids, dtype=torch.long).view(1, -1)
print(my_input, end="")
for x in model.generate(context, tokenizer, max_new_tokens=300):
    print(x, end="", flush=True)
    time.sleep(0.05)

