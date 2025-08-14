from src.networks.rltm import RLTM
from src.utils.config import ModelParameters


def main(pickle_name):
    model_parameters = ModelParameters()
    local_model = RLTM(pickle_name, model_parameters=model_parameters)
    
if __name__ == '__main__':
    pickle_name = "src/datasets/pickles/20newsgroups_mwl3"
    main(pickle_name)
