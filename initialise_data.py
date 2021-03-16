import os
import argparse
import pickle
from data_loader.dataset import DataSet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset for experiment.')
parser.add_argument('--input_dir', type=str, required=True, help='Input Directory of the processed data')
parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='targets')
parser.add_argument('--batch_size', type=int, help='BAtch Size for training', default=128)
args = parser.parse_args()

bin_file_path = os.path.join(args.input_dir, 'processed_inf.bin')
if os.path.exists(bin_file_path):
    pass
else:
    dataset = DataSet(train_src=os.path.join(args.input_dir, 'train_GGNNinput.json'),
                        valid_src=os.path.join(args.input_dir, 'valid_GGNNinput.json'),
                        test_src=os.path.join(args.input_dir, 'test_GGNNinput.json'),
                        batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                        l_ident=args.label_tag, inf=True)
    out_file = open(bin_file_path, 'wb')
    pickle.dump(dataset, out_file)
    out_file.close()