import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm


class Dataset:
    """
    输入语料，形式如["句子1" "句子2" "句子3"]
    使用下标直接返回某一数据 无batchsize，即batchsize=训练数据个数
    """
    def __init__(self, corpus, context_size=2):
        self.idx2word = list()
        self.word2idx = dict()
        self.train_data_contexts = []
        self.train_data_targets = []
        all_words = set()
        for sentence in corpus:
            all_words.update([word for word in sentence.split()])
        self.idx2word = list(all_words)
        for i, word in enumerate(self.idx2word):
            self.word2idx[word] = i
        for sentence in tqdm(corpus, desc="processing data"):
            sentence = sentence.split()
            sentence = [self.word2idx[word] for word in sentence]
            for i in range(1, len(sentence) - 1):
                target = sentence[i]
                left=max(0,i-context_size);
                right=min(len(sentence),i+context_size)
                context = sentence[left:i] + sentence[i+1:right]
                for c in context:
                 self.train_data_contexts.append(c)#构造单词对
                 self.train_data_targets.append(target)
    
    def len_of_vocab(self):
        return len(self.idx2word)

    def getdata(self):
        inputs = torch.tensor(self.train_data_contexts)
        targets = torch.tensor(self.train_data_targets)
        return inputs, targets


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        output = self.output(embeds)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs

embedding_dim = 10
context_size = 2
num_epoch = 200

corpus = ["w a s d q w", 
        "q w e r m n l",
        "w o r s v n y f s d d t u",
        "q b b f z p m r x g f a a"]

dataset = Dataset(corpus, context_size)
nll_loss = nn.NLLLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SkipGramModel(dataset.len_of_vocab(), embedding_dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in tqdm(range(num_epoch), desc="Training"):
    inputs, targets = [x.to(device) for x in dataset.getdata()]
    optimizer.zero_grad()
    log_probs = model(inputs)
    loss = nll_loss(log_probs, targets)
    loss.backward()
    optimizer.step()
print(f"Loss: {loss.item():.2f}")


def save_pretrained(vocab, embeds, save_path):
    with open(save_path, "w") as writer:
        writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(vocab.idx2word):
            vec = " ".join(["{:.4f}".format(x) for x in embeds[idx]])
            writer.write(f"{token} {vec}\n")
    print(f"Pretrained embeddings saved to: {save_path}")

save_pretrained(dataset, model.embeddings.weight.data, "cbow.simple.vec")