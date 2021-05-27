import pickle
import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np



class GraphRecModel(nn.Module):
    def __init__(self, embed_dim, num_users, num_ratings, num_items):
        super(GraphRecModel, self).__init__()
        self.embed_dim = embed_dim
        self.u2e = nn.Embedding(num_users, embed_dim)
        self.v2e = nn.Embedding(num_items, embed_dim)
        
        self.conv = dglnn.HeteroGraphConv({
                            'social' : dglnn.SAGEConv(embed_dim, embed_dim, 'mean'),
                            'rates' : dglnn.SAGEConv(embed_dim, embed_dim, 'mean'),
                            'rated' : dglnn.SAGEConv(embed_dim, embed_dim, 'mean')},
                            aggregate='sum')
        
        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.w_uv1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)

        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()
        
    def forward(self, g):
        h = {'user':self.u2e(g.nodes('user')), 'item':self.v2e(g.nodes('item'))}
        h = self.conv(g,h)
        #print(g.number_of_edges())
        embeds_u = h['user']
        embeds_v = h['item']
        
        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        with g.local_scope():
            sub_g = dgl.edge_type_subgraph(g, [('user', 'rates', 'item')])
            sub_g.nodes['user'].data['hu'] = x_u
            sub_g.nodes['item'].data['hv'] = x_v
            
            sub_g.apply_edges(dgl.function.u_mul_v('hu','hv','h_e'))
            
            x_uv = sub_g.edata['h_e']

#         x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)

        scores = self.w_uv3(x)

        return scores.squeeze()
    
    def loss(self, g, labels_list):
        scores = self.forward(g)
        return self.criterion(scores, labels_list)
    

def train(model,train_heter_graph, train_r, optimizer, epoch, best_rmse, best_mae):
    model.train()
    optimizer.zero_grad()
    loss = model.loss(train_heter_graph, train_r)
    loss.backward()
    optimizer.step()
    return 0

def test(model, test_heter_graph, test_r):
    model.eval()
    
    with torch.no_grad():
        val_output = model.forward(test_heter_graph).numpy()
    
        expected_rmse = np.sqrt(mean_squared_error(val_output, test_r))
        mae = mean_absolute_error(val_output, test_r)
        
    return expected_rmse, mae


if __name__ == '__main__':
    data_file = open("./data/toy_dataset.pickle", 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
        data_file)

    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    train_r = torch.tensor(train_r)
    test_r = torch.tensor(test_r)

    uu_src = []
    uu_dst = []
    for key, values in social_adj_lists.items():
        for value in values:
            uu_src.append(key)
            uu_dst.append(value)

    train_data_dict = {('user', 'social', 'user'): (uu_src,uu_dst),
                       ('user', 'rates', 'item'): (train_u, train_v),
                       ('item', 'rated', 'user'): (train_v, train_u)}
    test_data_dict = {('user', 'social', 'user'): (uu_src,uu_dst),
                      ('user', 'rates', 'item'): (test_u, test_v),
                      ('item', 'rated', 'user'): (test_v, test_u)}
    num_nodes_dict = {'user': num_users, 'item': num_items}

    train_heter_g = dgl.heterograph(train_data_dict, num_nodes_dict=num_nodes_dict)
    test_heter_g = dgl.heterograph(test_data_dict, num_nodes_dict=num_nodes_dict)

    model = GraphRecModel(64, num_users, num_ratings, num_items)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1,600+1):

        train(model,train_heter_g, train_r, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, mae = test(model,test_heter_g, test_r)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("[epoch %d]rmse: %.4f, mae:%.4f " % (epoch, expected_rmse, mae))

        if endure_count > 50:
            #print("Best case -- rmse: %.4f, mae:%.4f " % ( best_rmse, best_mae))
            break
        
    print("Best case -- rmse: %.4f, mae:%.4f " % ( best_rmse, best_mae))


