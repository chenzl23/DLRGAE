import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import random
from DataLoader import load_data
from paraparser import parameter_parser
from utils import tab_printer
from model import DLRGAE
from train import train

def main():
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    args = parameter_parser()
    args.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tab_printer(args)

    graph = load_data(args)
    number_class = torch.unique(graph.y).shape[0]  
    args.num_class = number_class
    processed_dir = os.path.join(os.path.join(os.path.join("../data",args.dataset_name), args.dataset_name), "processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    print(f"Data statistics:  #features {graph.x.size(1)}, #nodes {graph.x.size(0)}")

    graph = graph.to(device)

    input_channels = graph.x.size(1)
    output_channels = len(torch.unique(graph.y))

    model = DLRGAE(args, input_channels, output_channels, graph.x.size(0)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=5e-4)

    train(model, optimizer, graph, args)

if __name__ == "__main__":
    main()
