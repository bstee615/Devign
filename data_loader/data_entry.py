import dgl
import torch


class DataEntry:
    def __init__(self, get_edge_type_number, num_nodes, features, edges, target, filename):
        self.num_nodes = num_nodes
        self.target = target
        self.features = torch.FloatTensor(features)
        self.filename = filename

        self.graph = dgl.DGLGraph()
        self.features = torch.FloatTensor(features)
        self.graph.add_nodes(self.num_nodes, data={'features': self.features})
        for s, _type, t in edges:
            etype_number = get_edge_type_number(_type)
            self.graph.add_edge(s, t, data={'etype': torch.LongTensor([etype_number])})