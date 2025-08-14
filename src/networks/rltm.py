import os
import random
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from src.evaluation.plotting import Outputs, save_training_outputs
from src.utils.config import get_scheduler, step_scheduler
from src.config.config import DEVICE
from src.networks.decoder_network import DecoderNetwork
from src.datasets.dataset import get_datapipe


logger = logging.getLogger(__name__)


class RLTM:
    def __init__(self, pickle_name, model_parameters):
        self.model = DecoderNetwork(model_parameters)
        self.train_dataset = get_datapipe(pickle_name, 'train',
                                    frozen_embeddings=model_parameters.frozen_embeddings,
                                    sbert_model=model_parameters.sbert_model,
                                    hugface_model=model_parameters.hugface_model,
                                    max_length=model_parameters.max_length)
    
        self.model_parameters = model_parameters

        self.__train_local_model()       

    def _train_minibatch(self, model, batch, device, epoch, it):
        """Train a single minibatch and return output dict."""
        with torch.set_grad_enabled(True):
            # move batch tensors to device if dict
            if isinstance(batch, dict):
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            output_dict = model(batch)
            loss = output_dict['loss']
                
            return output_dict, loss
    
    def train_epoch_local_model(self, model, optimizer, scheduler, device, epoch):
        """Train one complete epoch and return output dict."""
        model.train(True)
        data = self.train_dataset 

        if data is None:
            return None
        
        if self.model_parameters.deterministic_dataloader:
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2 ** 32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            generator = torch.Generator()
            generator.manual_seed(self.model_parameters.seed)
            worker_init_fn = seed_worker
        else:
            worker_init_fn = None
            generator = None

        loader = DataLoader(data, pin_memory=False,
                            batch_size=self.model_parameters.batch_size,
                            num_workers=self.model_parameters.num_workers,
                            worker_init_fn=worker_init_fn,
                            generator=generator)

        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, batch in pbar:
            output_dict, loss = self._train_minibatch(model, batch, device, epoch, it)
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            if self.model_parameters.grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.model_parameters.grad_norm_clip)
            optimizer.step()

            lr = optimizer.param_groups[0]['lr']
            pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

        if self.model_parameters.lr_decay:
            step_scheduler(scheduler, output_dict)
    
    def __train_local_model(self):
        """Main training loop for local model."""
        model, model_parameters = self.model, self.model_parameters
        device = torch.device(DEVICE)
        model = model.to(device)

        optimizer = model.configure_optimizers(model_parameters)
        scheduler = None
        if model_parameters.lr_decay:
            scheduler = get_scheduler(optimizer)

        for epoch in range(model_parameters.max_epochs):
            self.train_epoch_local_model(model, optimizer, scheduler, device, epoch)
    