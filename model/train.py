import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import torch
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool as gap, global_max_pool as gmp
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


voting = pd.read_csv("../pygcn/data/BX-Book-Ratings.csv",
                     sep=';', encoding="ISO-8859-1")
books = pd.read_csv("../pygcn/data/BX_Books.csv",
                    sep=';', encoding="ISO-8859-1")
users = pd.read_csv("../pygcn/data/BX-Users.csv",
                    sep=';', encoding="ISO-8859-1")

book_id_encoder = LabelEncoder()
book_id_encoder.fit(pd.concat([books['ISBN'], voting['ISBN']]))
books['ISBN'] = book_id_encoder.transform(books['ISBN'])
voting['ISBN'] = book_id_encoder.transform(voting['ISBN'])

book_title_encoder = LabelEncoder()
books['Book-Title'] = book_title_encoder.fit_transform(books['Book-Title'])

book_publisher_encoder = LabelEncoder()
books['Publisher'] = book_publisher_encoder.fit_transform(
    books['Publisher'])

user_location_encoder = LabelEncoder()
users['Location'] = user_location_encoder.fit_transform(users['Location'])

users['Age'].fillna(0, inplace=True)

books['Year-Of-Publication'].fillna(0, inplace=True)

# print(voting.head())
# voting_encoder = OrdinalEncoder()
# voting['Book-Rating'] = voting_encoder.fit_transform(voting['Book-Rating'])


class BookDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(BookDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        grouped = voting.groupby('User-ID')
        for user_id, group in tqdm(grouped):
            user_book_id = LabelEncoder().fit_transform(group['ISBN'])
            group = group.reset_index(drop=True)
            group['user_book_id'] = user_book_id
            # print("\nuser book id: ", type(user_book_id))
            # print(group)

            node_features = group.loc[group['User-ID'] == user_id,
                                      ['user_book_id', 'ISBN']].sort_values('user_book_id')["ISBN"].values
            # print(node_features)

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group['user_book_id'].values[1:]
            source_nodes = group['user_book_id'].values[:-1]

            edge_index = torch.tensor(
                [source_nodes, target_nodes], dtype=torch.long)
            x = node_features
            y = torch.IntTensor([group['Book-Rating'].values[0]])
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


dataset = BookDataset("")
loader = DataLoader(dataset, batch_size=512, shuffle=True)

train_dataset = dataset[:round(len(dataset) * 0.8)]
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

valid_dataset = dataset[round(len(dataset) * 0.8): round(len(dataset) * 0.9)]
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True)

test_dataset = dataset[round(len(dataset) * 0.9): len(dataset)]
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True)


embed_dim = 128


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GCNConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GCNConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=voting['ISBN'].max() + 1, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)

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
        loss = crit(output, label)
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

torch.save(model.state_dict(), "weight.pt")
