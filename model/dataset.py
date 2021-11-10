import torch
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm

class BookDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(BookDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['BX-Book-Ratings.csv', 'BX-Users.csv', 'BX_Books.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def preprocessing(self):
        self.voting = pd.read_csv(self.raw_paths[0],
                     sep=';', encoding="ISO-8859-1")
        self.books = pd.read_csv(self.raw_paths[2],
                            sep=';', encoding="ISO-8859-1")
        self.users = pd.read_csv(self.raw_paths[1],
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

    def process(self):
        self.preprocessing()

        data_list = []
        self.voting = self.voting.merge(self.books, on="ISBN")
        self.voting = self.voting.merge(self.users, on="User-ID")

        grouped = self.voting.groupby('User-ID')
        for user_id, group in tqdm(grouped):
            user_book_id = LabelEncoder().fit_transform(group['ISBN'])
            group = group.reset_index(drop=True)
            group['user_book_id'] = user_book_id
            # print("\nuser book id: ", type(user_book_id))
            # print('group', group)

            node_features = group.loc[group['User-ID'] == user_id,
                                      ['user_book_id', 'ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Age', 'Location']].sort_values('user_book_id')[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Age', 'Location']].drop_duplicates().values
            print("node_features ", node_features)
            print("group ", group)
            node_features = torch.LongTensor(node_features)
            target_nodes = group['user_book_id'].values[:]
            source_nodes = group['user_book_id'].values[:]
            print("node ", source_nodes, target_nodes)
            edge_index = torch.tensor(
                [source_nodes, target_nodes], dtype=torch.long)
            x = node_features
            
            
            
            y = torch.IntTensor([group['Book-Rating'].values[0]])
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = BookDataset("./data")