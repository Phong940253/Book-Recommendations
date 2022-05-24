import torch
from torch_geometric.data import Data, DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class BookDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.preprocessing()
        super(BookDataset, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return ['BX-Book-Ratings.csv', 'BX-Users.csv', 'BX_Books.csv']

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(92001)]
        # range(len(self.voting))]

    @property
    def num_node_features(self):
        return 7

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def preprocessing(self):
        self.voting = pd.read_csv("./dataset/raw/BX-Book-Ratings.csv",
                     sep=';', encoding="ISO-8859-1")
        self.books = pd.read_csv("./dataset/raw/BX_Books.csv",
                            sep=';', encoding="ISO-8859-1")
        self.users = pd.read_csv("./dataset/raw/BX-Users.csv",
                            sep=';', encoding="ISO-8859-1")

        self.book_id_encoder = LabelEncoder()
        self.book_id_encoder.fit(pd.concat([self.books['ISBN'], self.voting['ISBN']]))
        self.books['ISBN'] = self.book_id_encoder.transform(self.books['ISBN'])
        self.voting['ISBN'] = self.book_id_encoder.transform(self.voting['ISBN'])

        self.book_title_encoder = LabelEncoder()
        self.books['Book-Title'] = self.book_title_encoder.fit_transform(self.books['Book-Title'])

        self.book_author_encoder = LabelEncoder()
        self.books['Book-Author'].fillna("", inplace=True)
        self.books['Book-Author'] = self.book_author_encoder.fit_transform(self.books['Book-Author'])

        self.book_publisher_encoder = LabelEncoder()
        self.books['Publisher'].fillna("", inplace=True)

        self.books['Publisher'] = self.book_publisher_encoder.fit_transform(
            self.books['Publisher'])

        self.user_location_encoder = LabelEncoder()
        self.users['Location'] = self.user_location_encoder.fit_transform(self.users['Location'])

        self.users['Age'].fillna(0, inplace=True)

        self.books['Year-Of-Publication'].fillna(0, inplace=True)
        # Read data into huge `Data` list.
        self.voting = self.voting.merge(self.books, on="ISBN")
        self.voting = self.voting.merge(self.users, on="User-ID")

        self.grouped = self.voting.groupby('User-ID')

    def build_edge_idx(self, num_nodes):
      # Initialize edge index matrix
      E = torch.zeros((2, num_nodes * (num_nodes - 1)), dtype=torch.long)
      
      # Populate 1st row
      for node in range(num_nodes):
          for neighbor in range(num_nodes - 1):
              E[0, node * (num_nodes - 1) + neighbor] = node

      # Populate 2nd row
      neighbors = []
      for node in range(num_nodes):
          neighbors.append(list(np.arange(node)) + list(np.arange(node+1, num_nodes)))
      E[1, :] = torch.LongTensor([item for sublist in neighbors for item in sublist])
      
      return E

    def process(self):
        self.preprocessing()

        data_list = []
        
        iters = 0
        for user_id, group in tqdm(self.grouped):
            if len(group['ISBN']) > 1000:
              continue
            sub_item_encoder = LabelEncoder()
            user_book_id = sub_item_encoder.fit_transform(group['ISBN'])
            group = group.reset_index(drop=True)
            group['user_book_id'] = user_book_id
            # print("\nuser book id: ", type(user_book_id))
            # print('group', group)

            node_features = group.loc[group['User-ID'] == user_id,
                                      ['user_book_id', 'ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Age', 'Location']].sort_values('user_book_id')[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Age', 'Location']].drop_duplicates().values
            # print("node_features ", node_features)
              # print("group ", group)
            node_features = torch.LongTensor(node_features)
            # target_nodes = group['user_book_id'].values[:]
            # source_nodes = group['user_book_id'].values[:]
            # print("node ", source_nodes, target_nodes)
            # edge_index = torch.tensor(
            #     [source_nodes, target_nodes], dtype=torch.long)
            edge_index = self.build_edge_idx(len(user_book_id))
            # print(edge_index.shape)
            # print(len(group['ISBN']))
            x = node_features
            print(edge_index, node_features)
            
            y = torch.IntTensor([group['Book-Rating'].values[0]])
            data = Data(x=x, edge_index=edge_index, y=y)
            torch.save(data, os.path.join(self.processed_dir, f'data_{iters}.pt'))
            iters +=1
        print(iters)
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
    def len(self):
      return 92001
      # return len(self.voting)

    def get(self, idx):
      return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))   

# dataset = BookDataset("./data")