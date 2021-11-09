import torch
from torch_geometric.data import Data, DataLoader, Dataset

class BookDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(BookDataset, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['BX-Book-Ratings.csv', 'BX-Users.cvs', 'BX_Books.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        voting = pd.read_csv(self.raw_paths[0],
                     sep=';', encoding="ISO-8859-1")
        books = pd.read_csv(self.raw_paths[0],
                            sep=';', encoding="ISO-8859-1")
        users = pd.read_csv(self.raw_paths[0],
                            sep=';', encoding="ISO-8859-1")

        book_id_encoder = LabelEncoder()
        book_id_encoder.fit(pd.concat([books['ISBN'], voting['ISBN']]))
        books['ISBN'] = book_id_encoder.transform(books['ISBN'])
        voting['ISBN'] = book_id_encoder.transform(voting['ISBN'])

        book_title_encoder = LabelEncoder()
        books['Book-Title'] = book_title_encoder.fit_transform(books['Book-Title'])

        book_publisher_encoder = LabelEncoder()
        books['Publisher'].fillna("", inplace=True)

        books['Publisher'] = book_publisher_encoder.fit_transform(
            books['Publisher'])

        user_location_encoder = LabelEncoder()
        users['Location'] = user_location_encoder.fit_transform(users['Location'])

        users['Age'].fillna(0, inplace=True)

        books['Year-Of-Publication'].fillna(0, inplace=True)
        # Read data into huge `Data` list.
        

        data_list = []
        global voting
        book_voting = voting.merge(books, on="ISBN")
        user_voting = book_voting.merge(users, on="User-ID")

        grouped = user_voting.groupby('User-ID')
        for user_id, group in tqdm(grouped):
            user_book_id = LabelEncoder().fit_transform(group['ISBN'])
            group = group.reset_index(drop=True)
            group['user_book_id'] = user_book_id
            # print("\nuser book id: ", type(user_book_id))
            # print(group)

            node_features = group.loc[group['User-ID'] == user_id,
                                      ['user_book_id', 'ISBN']].sort_values('user_book_id').ISBN.drop_duplicates().values
            print("node_features ", node_features)
            print("group ", group)
            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group['user_book_id'].values[1:]
            source_nodes = group['user_book_id'].values[:-1]
            print("node ", target_nodes, source_nodes)
            edge_index = torch.tensor(
                [source_nodes, target_nodes], dtype=torch.long)
            x = node_features
            y = torch.IntTensor([group['Book-Rating'].values[0]])
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])