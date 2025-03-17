import torch as t
import torch.nn.functional as F


class Simple(t.nn.Module):
    def __init__(self, vocab_size, max_seq_len, embedding_dim):
        super().__init__()
        self.embedding = t.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.fc1 = t.nn.Linear(max_seq_len * embedding_dim, 1024)
        self.fc2 = t.nn.Linear(1024, 64)
        self.logits = t.nn.Linear(64, 1)

    def forward(self, contents):
        batch_size = contents.shape[0]
        x = self.embedding(contents)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.logits(x).squeeze(dim=1)
