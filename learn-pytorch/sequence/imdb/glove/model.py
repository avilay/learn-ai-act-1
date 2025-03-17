import torch as t
import torch.nn.functional as F


class SimpleGlove(t.nn.Module):
    def __init__(self, imdb_vecs, max_seq_len):
        super().__init__()
        self.embedding = t.nn.Embedding.from_pretrained(imdb_vecs)
        embedding_dim = imdb_vecs[0].shape[0]
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
