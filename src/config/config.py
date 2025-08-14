import torch
import torch.nn as nn
from src.datasets.dataset import get_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelParameters:
    def __init__(self):
        # TRAINING CONFIGURATION
        # Distributed data parallel
        self.master_address = 'localhost'
        self.master_port = '12355'
        
        # Choose backend based on DEVICE
        self.backend = 'nccl' if DEVICE == 'cuda' else 'gloo'
        self.torch_distributed_debug = 'OFF'
        self.find_unused_parameters = False
        
        # Optimizer
        self.learning_rate = 3e-4
        self.betas = (0.9, 0.999)
        
        # Weight decay (on whitelist; not on blacklist)
        self.weight_decay = 0.01
        self.whitelist_weight_modules = (nn.Linear,)
        self.blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        self.whitelist_weight_names = ()
        self.blacklist_weight_names = ('prior_mean', 'prior_variance', 'beta')
        
        # Dataloader
        self.batch_size = 1024
        self.num_workers = 0
        self.deterministic_dataloader = True
        self.seed = 0
        
        # Training
        self.max_epochs = 1000
        self.grad_norm_clip = 1.0
        self.lr_decay = True
        self.ckpt_path = 'src/networks/ckpt'
        self.experiment_name = '20newsgroups_mwl3'
        self.experiment_num = 0
        self.check_experiment_path = True
        self.save_outputs = False
        self.save_pickle_every = 0
        
        
        # MODEL CONFIGURATION
        self.n_components = 20  # Number of topics
        
        # Decoder network
        self.input_size = 2000  # Vocabulary size
        self.input_type = 'bert'
        self.decoder_dropout = 0.2  # policy / theta dropout
        self.initialization = 'normal'
        self.normalization = 'layer'
        self.affine = False
        self.loss_type = 'rl'
        self.lda_type = 'prodlda'
        self.theta_softmax = False
        
        # Inference network
        self.frozen_embeddings = True
        self.sbert_model = 'all-MiniLM-L6-v2'
        self.hugface_model = 'sentence-transformers/all-MiniLM-L6-v2'
        self.bert_size = 384
        self.max_length = 256
        self.hiddens = (128, 128)
        self.activation = nn.GELU()
        self.inference_dropout = 0.2
        self.parameter_noise = False
        
        # Other
        self.prior = 'laplace'
        self.trainable_prior = True
        self.kl_mult = 1.0
        self.entropy_mult = 0.0
        self.topk = [10]
        self.sparse_corpus_bow = None
        self.pickle_name = 'src/datasets/pickles/20newsgroups_mwl3'  # Don't include ".pkl"
        self.get_sparse_corpus_bow()
        
    def get_sparse_corpus_bow(self):
        dataset_save_dict = get_dataset(self.pickle_name)
        self.sparse_corpus_bow = dataset_save_dict['sparse_corpus_bow']