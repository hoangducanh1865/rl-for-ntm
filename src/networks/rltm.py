import os
import random
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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

        if model_parameters.save_outputs:
            self.outputs = Outputs(model_parameters.experiment_name,
                                   model_parameters.experiment_num,
                                   model_parameters.check_experiment_path)
        self.__train_local_model(0)

    def setup(self, rank):
        os.environ['MASTER_ADDR'] = self.model_parameters.master_address
        os.environ['MASTER_PORT'] = self.model_parameters.master_port
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = self.model_parameters.torch_distributed_debug

        # initialize the process group
        dist.init_process_group(self.model_parameters.backend,
                                rank=rank, world_size=self.model_parameters.world_size)        

    def _train_minibatch(self, model, batch, device, epoch, it, rank):
        """Train a single minibatch and return output dict."""
        with torch.set_grad_enabled(True):
            # move batch tensors to device if dict
            if isinstance(batch, dict):
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            output_dict = model(batch)
            loss = output_dict['loss']
            
            if self.model_parameters.save_outputs:
                self.outputs.save_step(output_dict, 'train', epoch, it)
                
            return output_dict, loss
    
    def train_epoch_local_model(self, model, optimizer, scheduler, device, epoch, rank):
        """Train one complete epoch and return output dict."""
        model.train(True)
        data = self.train_dataset 

        if data is None:
            return None

        if self.model_parameters.world_size > 1:
            distributed_sampler = DistributedSampler(data, shuffle=True)
        else:
            distributed_sampler = None

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
                            generator=generator,
                            sampler=distributed_sampler)

        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader)) if (rank == 0) else enumerate(loader)
        for it, batch in pbar:
            output_dict, loss = self._train_minibatch(model, batch, device, epoch, it, rank)
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            if self.model_parameters.grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.model_parameters.grad_norm_clip)
            optimizer.step()

            if rank == 0:
                lr = optimizer.param_groups[0]['lr']
                pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

        if self.model_parameters.lr_decay:
            step_scheduler(scheduler, output_dict)

        return output_dict
    
    def __train_local_model(self, rank):
        """Main training loop for local model."""
        model, model_parameters = self.model, self.model_parameters
        use_ddp = (model_parameters.world_size > 1)
        if use_ddp:
            self.setup(rank)
        device = torch.device(DEVICE)
        model = model.to(device)
        if use_ddp and DEVICE == 'cuda':
            model = DDP(model, device_ids=[rank], find_unused_parameters=model_parameters.find_unused_parameters)
        elif use_ddp:
            model = DDP(model, find_unused_parameters=model_parameters.find_unused_parameters)

        raw_model = model.module if hasattr(model, "module") else model
        optimizer = raw_model.configure_optimizers(model_parameters)
        scheduler = None
        if model_parameters.lr_decay:
            scheduler = get_scheduler(optimizer)

        for epoch in range(model_parameters.max_epochs):
            output_dict = self.train_epoch_local_model(model, optimizer, scheduler, device, epoch, rank)
            if model_parameters.save_outputs and output_dict is not None:
                self.outputs.save_epoch(output_dict, epoch)

        if model_parameters.save_outputs:
            if model_parameters.save_pickle_every > 0:
                self.outputs.save_pickle()
            save_training_outputs(model_parameters.experiment_name,
                                  model_parameters.experiment_num,
                                  self.outputs)

        if use_ddp:
            dist.destroy_process_group()
    