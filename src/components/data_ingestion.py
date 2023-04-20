import os
import sys
import h5py
import numpy as np
import torch
# import logging
import random
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset
import torch.nn as nn
from typing import Any, Tuple, Dict, List, Union
import math

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts', 'train.csv')
    test_data_path=os.path.join('artifacts', 'test.csv')
    raw_data_path=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.config_path = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)



if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()



Tensor = torch.Tensor
LongTensor = torch.LongTensor

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
logger = logging.getLogger(__name__)

Optimizer = torch.optim.Optimizer
Scheduler = torch.optim.lr_scheduler._LRScheduler

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    )

logger.setLevel(logging.DEBUG)

def gelu(x: Tensor) -> Tensor:
    """Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class Conv1D(nn.Module):
    """1D-convolutional layer (eqv to FCN) as defined by Radford et al. for OpenAI GPT 
    (and also used in GPT-2). Basically works like a linear layer but the weights are transposed.

    Note: 
        Code adopted from: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_utils.py

    Args:
        nf (int): The number of output features.
        nx (int): The number of input features.
    """
    def __init__(self, nf: int, nx: int) -> None:
        """Constructor
        """
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): [..., nx] input features

        Returns:
            Tensor: [..., nf] output features
        """
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x
    
#Generalization 
config = {
    "model": {
        "n_embd": 1568,
        "n_layer": AAAAA,
        "n_head": BBBBB,
        "activation_function": gelu,
        "dropout": 0.0,
        "seed": 12345,
        "path": "/bigdata/wonglab/syang159/CFD2/pytorch_particle_random_order/" + "IIIII",
        "path_result": "/bigdata/wonglab/syang159/CFD2/pytorch_particle_random_order/JJJJJ/Result"
    },
    "training": {
        "device": "cuda", # "cuda" or "cpu"
        "batch_size": FFFFF,
        "training_h5_file": "/bigdata/wonglab/syang159/CFD2/Kfold_data/training_data_HHHHH.hdf5",
        "n_ctx": CCCCC,
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
        "layer_norm_epsilon": 1e-05,
        "initializer_range": 0.01,
        "stride": DDDDD,
        "ndata": 2500,  #2500
        "num_epoch": 100, #100
        "learning_rate": LLLLL,
        "max_lr": MMMMM,
        #"T_0_for_cosine_annealing": EEEEE,
        #"T_mult_for_cosine_annealing": GGGGG,
        "scheduler_step_size": 100,
        "output_hidden_state": False,
        "output_attention": False,
        "use_cache": True,
        "max_length": 50,
        "min_length": 0,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 0.01,
        "Job_number": KKKKK,
        "MSE_threshold": 0.0065,
        "es_counter": 3
    },
    "validating": {
        "validating_h5_file": "/bigdata/wonglab/syang159/CFD2/Kfold_data/validating_data_HHHHH.hdf5",
        "batch_size": FFFFF,
        "block_size": 50,
        "stride": 3000,
        "ndata": 312,  #312
        "val_steps":1
    },
    "testing": {
        "testing_h5_file": "/bigdata/wonglab/syang159/CFD2/Kfold_data/testing_data.hdf5",
        "batch_size": FFFFF,
        "block_size": 50,
        "stride": 3000,
        "ndata": 313  #313
    }
}

def seed_torch(seed=config['model']['seed']):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
seed_torch()
g= torch.Generator()

#Load data
class DatasetReader(Dataset):
    def __init__(
        self,
        file_path: str,
        block_size: int,
        stride: int = 1,
        ndata: int = -1,
        eval: bool = False,
        **kwargs
    ):

        self.block_size = block_size
        self.stride = stride
        self.ndata = ndata
        self.eval = eval   
        self.examples = []
        self.states = []
        with h5py.File(file_path, "r") as f:
            self.load_data(f, **kwargs)  

    # TODO: modify the h5_file to include Boiler Headers Mesh, then modify load_data to apply the mesh as the embedding vectors. 
        # For each time step, the embedding vector is fixed, it will be used as a constraint to the trajectories. 
        # We can use the koopman technique to calculate the embedding vector using all timestep results.  
    def load_data(self, h5_file: h5py.File) -> None:

        # Iterate through stored time-series
        with h5_file as f:
            params0 = torch.Tensor(f['params'])
            pos_x = torch.Tensor(f['x'])
            pos_y = torch.Tensor(f['y'])
            pos_z = torch.Tensor(f['z'])
            for (p, x, y, z) in zip(params0, pos_x, pos_y, pos_z):
                data_series = torch.stack([x, y, z], dim=1).to(config["training"]["device"])
                data_series = data_series[:,:,torch.randperm(data_series.size(2), generator=g.manual_seed(config['model']['seed']))]

                p=p.to(config["training"]["device"])
                data_series1 = torch.cat([data_series, p.unsqueeze(-1) * torch.ones_like(data_series[:,:1])], dim=1)
                data_series1 = data_series1.view(data_series1.size(0),data_series1.size(1)*data_series1.size(2))
    
                # Stride over time-series
                for i in range(0, data_series1.size(0) - self.block_size + 1, self.stride):
                    
                    data_series0 = data_series1[i: i + self.block_size]  # .repeat(1, 4)
                    self.examples.append(data_series0)
    
                    if self.eval:
                        self.states.append(data_series[i: i+ self.block_size].cpu())
        
                
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        if self.eval:
            return {'inputs_x': self.examples[i][:1], 'labels': self.examples[i]}
        else:
            return {'inputs_x': self.examples[i][:-1], 'labels': self.examples[i][1:]}
    
class DataCollator:
    """
    Data collator used for training datasets. 
    Combines examples in a minibatch into one tensor.
    
    Args:
        examples (List[Dict[str, Tensor]]): List of training examples. An example
            should be a dictionary of tensors from the dataset.

        Returns:
            Dict[str, Tensor]: Minibatch dictionary of combined example data tensors
    """
    # Default collator
    def __call__(self, examples:List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        mini_batch = {}
        for key in examples[0].keys():
            mini_batch[key] = self._tensorize_batch([example[key] for example in examples])

        return mini_batch

    def _tensorize_batch(self, examples: List[Tensor]) -> Tensor:
        if not torch.is_tensor(examples[0]):
            return examples

        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)

        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            raise ValueError("Padding not currently supported for physics transformers")
            return

training_data = DatasetReader(
    config["training"]["training_h5_file"], 
    block_size=config["training"]["n_ctx"], 
    stride=config["training"]["stride"],
    ndata=config["training"]["ndata"], 
    )

validating_data = DatasetReader(
    config["validating"]["validating_h5_file"], 
    block_size=config["validating"]["block_size"], 
    stride=config["validating"]["stride"],
    ndata=config["validating"]["ndata"], 
    eval = True,
    )

training_loader = DataLoader(
    training_data,
    batch_size=config["training"]["batch_size"],
    sampler=RandomSampler(training_data),
    collate_fn=DataCollator(),
    drop_last=True,
)

validating_loader = DataLoader(
    validating_data,
    batch_size=config["validating"]["batch_size"],
    sampler=SequentialSampler(validating_data),
    collate_fn=DataCollator(),
    drop_last=True,
)


#Testing

testing_data = DatasetReader(
    config["testing"]["testing_h5_file"], 
    block_size=config["testing"]["block_size"], 
    stride=config["testing"]["stride"],
    ndata=config["testing"]["ndata"], 
    eval = True,
    )

testing_loader = DataLoader(
    testing_data,
    batch_size=config["testing"]["batch_size"],
    sampler=SequentialSampler(testing_data),
    collate_fn=DataCollator(),
    drop_last=True,
)