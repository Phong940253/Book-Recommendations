import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import torch
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool as gap, global_max_pool as gmp
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from dataset import BookDataset

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def merge_file(prefix_filename, num_files):
    with open(prefix_filename, 'wb+') as f:
        for idx in range(num_files):
            with open(prefix_filename + '_part_' + str(idx + 1), 'rb') as chunk_file:
                chunk = chunk_file.read()
                f.write(chunk)
# merge_file('./dataset/processed/data.pt', 32)

dataset = BookDataset(root="./dataset")
# for i in range(10):
#   print(dataset[i])
# loader = DataLoader(dataset, batch_size=512, shuffle=True)

train_dataset = dataset[:round(len(dataset) * 0.8)]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = dataset[round(len(dataset) * 0.8): round(len(dataset) * 0.9)]
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

test_dataset = dataset[round(len(dataset) * 0.9): len(dataset)]
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


embed_dim = 128
# voting = pd.read_csv(dataset.raw_paths[0],
#                      sep=';', encoding="ISO-8859-1")
# voting['ISBN'] = LabelEncoder().fit_transform(voting['ISBN'])

class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max', node_dim=-2) #  "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()
        
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # print("forward")
        
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # print(x.shape, x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # print("message")
        # x_j has shape [E, in_channels]

        x_j = self.lin(x_j)
        x_j = self.act(x_j)
        
        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # print("update")
        # print(aggr_out.shape, x.shape)
        new_embedding = torch.cat([aggr_out, x], dim=1)
        # print(new_embedding.shape)
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)
        
        return new_embedding

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=dataset.voting.ISBN.max() + 1, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(x)
        # print(edge_index)
        x = x[:, 0]
        x = self.item_embedding(x)
        x = x.squeeze(1)
        # print(x)
        # print(edge_index.shape)
        # print(x.shape, batch)
        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)
        return x


def evaluate(loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)


def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = crit(output.type(torch.FloatTensor), label.type(torch.FloatTensor))
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)



model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
crit = torch.nn.BCELoss()

for epoch in range(100):
    loss = train()
    train_acc = evaluate(train_loader)
    val_acc = evaluate(valid_loader)
    test_acc = evaluate(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.
          format(epoch, loss, train_acc, val_acc, test_acc))
