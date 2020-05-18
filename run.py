from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
from eval import eval
import torch
def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    simclr = SimCLR(dataset, config)
    simclr.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = simclr.model
    eval(model,'./data/',device,config)


if __name__ == "__main__":
    main()
