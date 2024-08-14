#import torch_geometric.transforms as T
from torch import manual_seed
from torch.nn import Module, ModuleList, Embedding
import torch.nn.functional as F 
from torch_geometric.nn import HeteroConv, Linear, GraphConv, global_mean_pool
from global_concat_pool import global_concat_pool

class model_1(Module):
    def __init__(self, num_players, num_teams, embedding_players, embedding_teams,
                 hidden_channels, out_channels, num_layers):
        super().__init__()
        manual_seed(12345)

        # embedding de jogadores
        self.player_emb = Embedding(num_players, 
                                    embedding_players)
        
        # embedding de times
        self.team_emb = Embedding(num_teams, 
                                  embedding_teams)


        # operadores de convolução
        self.convs = ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('players', 'plays', 'team'): GraphConv((-1, -1), 
                                                        hidden_channels[0]),
                ('team', 'rev_plays', 'players'): GraphConv((-1, -1), 
                                                            hidden_channels[0]),
            }, aggr='sum')
            self.convs.append(conv)

        # camda que decide as classes
        #self.lin = Linear(hidden_channels[0], out_channels)
        self.lin = Linear(-1, out_channels)

    def forward(self, data, batch):
        x_dict = {
                    "players": self.player_emb(data["players"].node_id),
                    "team": self.team_emb(data["team"].node_id)
                }
        edge_index_dict = data.edge_index_dict
        
        for conv in self.convs: 
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=0.1, training=self.training) for key, x in x_dict.items()}

        y = global_concat_pool(x_dict["team"], batch)
        
        y = F.softmax(self.lin(y))

        return y, x_dict["team"]
    
