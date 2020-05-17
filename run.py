from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
from eval import eval

def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    simclr = SimCLR(dataset, config)
    simclr.train()

    model = simclr.model
    eval(model,'./data/',config['device'])


if __name__ == "__main__":
    main()
