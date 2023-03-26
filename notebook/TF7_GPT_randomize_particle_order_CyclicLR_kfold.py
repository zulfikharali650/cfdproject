#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:07:38 2023

@author: user
"""

import numpy as np
import os
import h5py
import torch
import logging
import torch.optim as optim
from typing import Any, Tuple, Dict, List, Union
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
import random
import matplotlib.pyplot as plt


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

#Define model
class MLP(nn.Module):
    """Simple fully connected neural network layer.
    Includes activations function and dropout.

    Args:
        n_state (int): dimensionality of input features
        config (PhysConfig): Phys-transformer config object
    """
    def __init__(self, n_state: int) -> None:
        """Constructor 
        """
        super().__init__()
        nx = config["model"]["n_embd"]
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = config["model"]["activation_function"]
        self.dropout = nn.Dropout(config["training"]["resid_pdrop"])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): [B, T, n_state] input features

        Returns:
            Tensor: Output features
        """
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)
    
class MaskedAttention(nn.Module):
    """Masked self-attention module based on the Hugging face implementation
    https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py

    Args:
        nx (int): Dimensionality of feature vector
        n_ctx (int): Context length of the attention (TODO: Not needed with config object?)
        config (PhysConfig): Transformer config object
        scale (bool, optional): Scale the attention scores. Defaults to False.
        mask (str, optional): Attention mask type. Defaults to 'tril'.

    Raises:
        ValueError: Invalid mask type
    """
    def __init__(
        self, 
        nx: int, 
        n_ctx: int, 
        scale: bool  = False, 
        mask: str = 'tril'
    ) -> None:
        """Constructor
        """
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        assert n_state % config["model"]["n_head"] == 0
        
        # Create attention mask
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config["model"]["n_head"]
        self.split_size = n_state
        self.scale = scale

        # Conv1D are not PyTorch Conv1D
        # Conv1D(out_features, in_features)
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config["training"]["attn_pdrop"])
        self.resid_dropout = nn.Dropout(config["training"]["resid_pdrop"])

    def _attn(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor, 
        attention_mask: Tensor = None,
        head_mask: Tensor = None, 
        output_attentions: bool =False
    ) -> List[Tensor]:
        """Dot product attention calculation

        Args:
            q (Tensor): [batch, head, seq_length, head_features] query
            k (Tensor): [batch, head, head_features, seq_length] key
            v (Tensor): [batch, head, seq_length, head_features] value
            attention_mask (Tensor, optional): Optional defined attention mask. Defaults to None.
            head_mask (Tensor, optional): Optional attention value mask. Defaults to None.
            output_attentions (bool, optional): Output attention matrix. Defaults to False.

        Returns:
            List[Tensor]: Output consisting of output feature, attention tensor (if requested)
        """
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)

        nd, ns = w.size(-2), w.size(-1)
        mask = self.bias[:, :, ns - nd : ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask
        
        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x: Tensor) -> Tensor:
        """Merge attention heads

        Args:
            x (Tensor): [batch, head, seq_length, head_features] Input tensor

        Returns:
            Tensor: [batch, seq_length, head * head_features] Concatenated output tensor
        """
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k: bool = False) ->  Tensor:
        """Splits key, query or value tensor into separate heads.
        Dimensionality of output depends if tensor is a key.

        Args:
            x (Tensor): [batch, seq_length, nx] Input tensor
            k (bool): If input tensor is a key tensor

        Returns:
            Tensor: [batch, head, seq_length, head_features] Split features for query
            and value, [batch, head, seq_length, head_features] split feature for key
        """
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self, 
        x: Tensor, 
        layer_past: List[Tensor] = None, 
        attention_mask: Tensor =None, 
        head_mask: Tensor = None,
        use_cache: bool = False, 
        output_attentions: bool = False
    ) -> List[Tensor]:
        """Masked attention forward pass

        Args:
            x (Tensor): [batch, seq_length, nx] Input feature.
            layer_past (Tensor, optional): [2, batch, n_head, seq_length, nx] Precomputed self-attention vectors. Defaults to None.
            attention_mask (Tensor, optional): Optional defined attention mask. Applied before soft mask.
                 Defaults to None.
            head_mask (Tensor, optional): Optional attention value mask. Applied after softmax Defaults to None.
            use_cache (bool, optional): Return calculated key values or faster generation. Defaults to False.
            output_attentions (bool, optional): Return attention matrix. Defaults to False.

        Returns:
            List[Tensor]: Output consisting of output feature, key values (if requested), attention tensor (if requested)
        """
        x = self.c_attn(x) # x -> q, k, v
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True) # k=True for keys which transposes the last two dims
        value = self.split_heads(value)
        # Concat previous key and value tensors 
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)

class Block(nn.Module):
    """Transformer decoder block consisting of layer norm, masked self-attention,
    layer norm and fully connected layer.

    Args:
        n_ctx (int): contex length of block
        config (PhysConfig): Phys-transformer config object
        scale (bool, optional): Scaled self-attention calculation. Defaults to False.
    """
    def __init__(self, n_ctx: int, scale: bool = False) -> None:
        """Constructor
        """
        super().__init__()
        nx = config["model"]["n_embd"]
        self.ln_1 = nn.LayerNorm(nx, eps=config["training"]["layer_norm_epsilon"])
        self.attn = MaskedAttention(nx, n_ctx, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config["training"]["layer_norm_epsilon"])
        self.mlp = MLP(4 * nx)

    def forward(
        self, 
        x: Tensor, 
        layer_past: List[Tensor] = None, 
        attention_mask: LongTensor = None, 
        head_mask: LongTensor = None, 
        use_cache: bool = False, 
        output_attentions: bool = False,
    ) -> List[Tensor]:
        """Forward pass

        Args:
            x (Tensor): [B, T, n_state] input features
            layer_past ([type], optional): Past self-attention calculation. Defaults to None.
            attention_mask (LongTensor, optional): Attention mask. Defaults to None.
            head_mask (LongTensor, optional): Attention value. Defaults to None.
            use_cache (bool, optional): Store attention state (key values). Defaults to False.
            output_attentions (bool, optional): Return attention values. Defaults to False.

        Returns:
            List[Tensor]: List of output tensors
        """
        # Evaluate attention heads
        output_attn = self.attn.forward(
            self.ln_1(x),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions
        )

        a = output_attn[0] 
        # Residual connection 1
        x = x + a
        # FCNN
        m = self.mlp(self.ln_2(x))
        # Residual connection 2
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)
    
class TransformerBase(nn.Module):
    """Parent class for physical transformers
    """
    model_name: str = "transformer_model"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        # Save config in model
        

    def forward(self):
        pass

    def generate(self):
        pass

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=config["training"]["initializer_range"])
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _num_parameters(self) -> int:
        """Gets number of learnable parameters

        Returns:
            int: Number of parameters
        """
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count
    
    def save_model(self, save_directory: str, epoch: int = 0) -> None:
        """Saves transformer model to the specified directory.

        Args:
            save_directory (str): Folder to save file at
            epoch (int, optional): Epoch number to name model file. Defaults to 0.

        Raises:
            AssertionError: If provided directory is not valid.
        """
        if os.path.isfile(save_directory):
            raise AssertionError("Provided path ({}) should be a directory, not a file".format(save_directory))

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "{}{:d}.pth".format(self.model_name, epoch))
        # Save pytorch model to file
        torch.save(self.state_dict(prefix='transformer'), output_model_file)


    def load_model(self, file_or_path_directory: str, epoch: int = 0) -> None:
        """Load a transformer model from the specified file or path
        
        Args:
            file_or_path_directory (str): File or folder path to load state dictionary from.
            epoch (int, optional): Epoch of current model for file name, used if folder path is provided. Defaults to 0.
        
        Raises:
            FileNotFoundError: If provided file or directory could not be found.
        """
        if os.path.isfile(file_or_path_directory):
            logger.info('Loading embedding model from file: {}'.format(file_or_path_directory))
            self.load_state_dict(torch.load(file_or_path_directory, map_location=lambda storage, loc: storage))
        elif  os.path.isdir(file_or_path_directory):
            file_path = os.path.join(file_or_path_directory, "{}{:d}.pth".format(self.model_name, epoch))
            logger.info('Loading embedding model from file: {}'.format(file_path))
            self.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
        else:
            raise FileNotFoundError("Provided path or file ({}) does not exist".format(file_or_path_directory))
            
class TransformerTrain(TransformerBase):
    """Model head for training the physics transformer base.

    Args:
        config (PhysConfig): Phys-transformer config object
        transformer_model (PhysformerBase): Initialized transformer model
    """
    def __init__(self, transformer_model: TransformerBase = None) -> None:
        """Constructor
        """
        super().__init__()
        self.transformer = transformer_model
        self.transformer.apply(self._init_weights)

    def forward(
        self,
        inputs_x: Tensor,
        labels: Tensor,
        **kwargs
    ) -> Tuple[Union[float, Tensor]]:
        """Forward method for this head calculates the MSE between the predicted time-series and target embeddings
        This head allows for easy distribution to multiple GPUs and CPUs. See transformer 

        Args:
            inputs_embeds (Tensor): [B, T, n_embed] Input features
            labels_embeds (Tensor): [B, T, n_embed] Target output features
            **kwargs (optional): Additional tensformer forward pass arguments

        Returns:
            Tuple[Union[float, Tensor]]: mse loss, last hidden state, (present attention state), 
            (all hidden_states), (attention scores)
        """
        outputs = self.transformer.forward(
            inputs_x=inputs_x,
            **kwargs
        )
        # If label embeddings are provided, compute loss
        if labels is not None:
            hidden_states = outputs[0]

            # Flatten the tokens
            loss_fct = nn.MSELoss()
            loss = loss_fct(hidden_states, labels)

            # loss = loss+ loss_fct(shift_hidden[:,:3], shift_labels[:,:3])
            outputs = (loss,) + (hidden_states, labels,) + outputs[1:]

        return outputs # (loss), last hidden state, (presents), (all hidden_states), (attentions)

    def evaluate(
        self,
        inputs_x: Tensor,
        labels: Tensor,
        **kwargs
    ) -> Tuple[Union[float, Tensor]]:
        """Generate a time-series prediction using the transformer and calc MSE error.

        Args:
            inputs_embeds (Tensor): [B, 1, n_embed] Starting input feature(s)
            labels_embeds (Tensor): [B, T, n_embed] Target output features
            **kwargs (optional): Additional tensformer forward pass arguments

        Returns:
            Tuple[Union[float, Tensor]]: mse loss, last hidden state, (present attention state), 
            (all hidden_states), (attention scores)
        """

        max_length = labels.size(1)

        outputs = self.transformer.generate(
            inputs_x=inputs_x,
            max_length = max_length,
            **kwargs
        )
        predicted = outputs[0]

        # Flatten the tokens
        err_fct = nn.MSELoss()
        error = err_fct(predicted, labels)

        outputs = (error,) + (predicted, labels,) + outputs[1:]

        return outputs

    def generate(self, *args, **kwargs):
        """
        Generate call is just the forward call of the transformer
        """
        return self.transformer.generate(*args, **kwargs)

class Time_Series_Generation:
    """Class containing generative functions for transformers
    """
    def prepare_inputs_for_generation(
        self, 
        inputs_x: Tensor,
        position_ids: Tensor = None,
        prop_embeds: Tensor = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """Prepares input features for prediction

        Args:
            inputs_features (Dict[str, Tensor]): Input feature tensors
            that are being generated.

        Returns:
            Dict[str, Tensor]: Dictionary of model inputs
        """
        inputs_features = {
            "inputs_x": inputs_x,
            "position_ids": position_ids,
            "prop_embeds": prop_embeds }
        inputs = {}

        for k, v in inputs_features.items():
            if isinstance(v, torch.Tensor):
                # Make sure all embeddings are of equal and proper length
                inputs[k] = v[:, -config["training"]["n_ctx"]:]

        if "past" in kwargs.keys():
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v[:, -1].unsqueeze(1)

        return {**inputs, **kwargs}

    def _use_cache(self, outputs, use_cache):
        """During generation, decide whether to pass the `past` variable to the next forward pass."""
        if len(outputs) <= 1 or use_cache is False:
            return False
        if hasattr(self.config, "mem_len") and self.config.mem_len == 0:
            return False
        return True

    @torch.no_grad()
    def generate(
        self,
        inputs_x: Tensor,
        position_ids: Tensor = None,
        prop_embeds: Tensor = None,
        max_length: int = None,
        attention_mask: LongTensor = None,
        use_cache: bool = False,
        **model_specific_kwargs
    ) -> Tuple[Tensor]:
        """Generated a predicted sequence of features

        Args:
            inputs_embeds (Tensor): [batch, seq, n_embed] Input feature tensor
            position_ids (Tensor, optional): [seq, n_embed] Position tensor. Defaults to None.
            prop_embeds (Tensor, optional): [batch, seq, n_embed] Property tensor. Defaults to None.
            max_length (int, optional): Length of time series to predict. Defaults to None.
            attention_mask (LongTensor, optional): Manual attention mask. Defaults to None.
            use_cache (bool, optional): Cache past transformer states for faster generation. Defaults to False.

        Returns:
            Tuple[Tensor]: [batch, max_length, n_embed] Predicted feature tensor, additional optional transformer outputs.
        """
        max_length = max_length if max_length is not None else self.config.max_length
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."

        output = self._generate_time_series(
            inputs_x,
            position_ids,
            prop_embeds,
            max_length=max_length,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **model_specific_kwargs,
        )

        return output

    def _generate_time_series(
        self,
        inputs_x: Tensor,
        position_ids: Tensor,
        prop_embeds: Tensor,
        max_length: int,
        use_cache: bool = None,
        **model_specific_kwargs
    ) -> Tuple[Tensor]:
        """Function that calls model forward to predict 

        Args:
            inputs_embeds (Tensor): [batch, seq, n_embed] Input feature tensor
            position_ids (Tensor, optional): [seq, n_embed] Position tensor. Defaults to None.
            prop_embeds (Tensor, optional): [batch, seq, n_embed] Property tensor. Defaults to None.
            max_length (int, optional): Length of time series to predict. Defaults to None.
            attention_mask (LongTensor, optional): Manual attention mask. Defaults to None.
            use_cache (bool, optional): [description]. Defaults to None.

        Returns:
            Tuple[Tensor]: [batch, max_length, n_embed] Predicted feature tensor, additional optional transformer outputs.
        """
        past = None

        cur_len = inputs_x.shape[1]
        assert (
            cur_len < max_length
        ), f"The input context is {cur_len}, but `max_length` is only {max_length}. Please make sure that `max_length` larger than the input"

        while cur_len < max_length:
            # Prepare inputs for transformer
            model_inputs = self.prepare_inputs_for_generation(
                inputs_x, 
                position_ids, 
                prop_embeds, 
                use_cache=use_cache, 
                past = past,
                **model_specific_kwargs,
            )

            outputs = self.forward(**model_inputs)

            next_output = outputs[0][:,-1:]

            if self._use_cache(outputs, use_cache):
                past = [output[:, :, :, -(self.config.n_ctx-1):] for output in outputs[1]]

            # add past output embedding and increase length by one
            inputs_x = torch.cat([inputs_x, next_output], dim=1)
            cur_len = cur_len + 1

            # If number of time-steps has surpassed model capacity, start dropping
            # the earliest time-step from the past states
            # if(cur_len > self.config.n_ctx):
                # Dim [keys/query, batch, heads, tsteps, n_embed]
                # past = tuple(attention_state[:,:,:,1:] for attention_state in past)

        return (inputs_x, ) + outputs[1:]
    
class PhysformerGPT2(Time_Series_Generation, TransformerBase):
    """Transformer decoder model for modeling physics

    Args:
            config (PhysConfig): Phys-transformer config object
            model_name (str, optional): Model name. Defaults to None.
    """
    def __init__(self) -> None:
        """Constructor        
        """
        TransformerBase.__init__(self)
        self.output_hidden_states = config["training"]["output_hidden_state"]

        self.drop = nn.Dropout(config["training"]["embd_pdrop"])
        self.h = nn.ModuleList([Block(config["training"]["n_ctx"], scale=True) for _ in range(config["model"]["n_layer"])])
        self.ln_f = nn.LayerNorm(config["model"]["n_embd"], eps=config["training"]["layer_norm_epsilon"])
        self.mlp_f = nn.Linear(config["model"]["n_embd"], config["model"]["n_embd"])
        #self.wpe = nn.Embedding(config["training"]["n_ctx"], config["model"]["n_embd"])
        self.apply(self._init_weights)

        self.n_embd = config["model"]["n_embd"]

        logger.info('Number of parameters: {}'.format(self._num_parameters()))
            
    def forward(
        self,
        inputs_x: Tensor,
        position_ids: Tensor = None,
        past: List[List[Tensor]] = None,
        attention_mask: LongTensor = None,
        head_mask: LongTensor = None,
        use_cache: bool = True,
        output_attentions: bool = False
    ) -> List[Tensor]:
        """Forward pass

        Note: Attention masks are not properly implemented presently and will likely not work.

        Args:
            inputs_embeds (Tensor): [B, T, n_embed] Input features
            position_ids (Tensor, optional): [T, n_embed] Manually specify position ids. Defaults to None.
            prop_embeds (Tensor, optional): [B, T, n_embed] Optional property feature. Defaults to None.
            past (List[List[Tensor]], optional): Transformer past state. Defaults to None.
            attention_mask (LongTensor, optional): [B, T] Sequence attention mask. Defaults to None.
            head_mask (LongTensor, optional): Attention value mask. Defaults to None.
            use_cache (bool, optional): Return attention states (keys). Defaults to True.
            output_attentions (bool, optional): Return attention scores. Defaults to False.

        Returns:
            List[Tensor]:  Output features, attention state (if requested), 
            hidden states of all layers (if requested), attention tensor (if requested)
        """

        # Input embeddings
        input_shape = inputs_x.size()[:-1]
        batch_size = inputs_x.shape[0]

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
            
        if position_ids is None:
            device = config["training"]["device"]
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.float, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1]).repeat(inputs_x.size(0),1)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Set mask to 0 for positions we want to attend and -10000 for ones we do not
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Function embeddings proposed in original transformer paper
        # http://papers.nips.cc/paper/7181-attention-is-all-you-need
        position_embeds = torch.zeros_like(inputs_x)
        i = torch.arange(0, config["model"]["n_embd"] // 2, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
        position_embeds[:, :, ::2] = torch.sin(position_ids.unsqueeze(-1) / 10000 ** (2 * i / config["model"]["n_embd"]))
        i = i[:, :, config["model"]["n_embd"] % 2]
        position_embeds[:, :, 1::2] = torch.cos(position_ids.unsqueeze(-1) / 10000 ** (2 * i / config["model"]["n_embd"]))
        
        # Combine input embedding, position embeding and prop embeddings
        hidden_states = inputs_x + position_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        # Loop through transformer self-attention layers
        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.mlp_f(self.ln_f(hidden_states))

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
            
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)

transformer = PhysformerGPT2()
model = TransformerTrain(transformer)

#Define Trainer
def set_seed(seed: int) -> None:
    """Set random seed

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Trainer:

    """Generalized trainer for physics transformer models

    Args:
        model (PhysformerTrain): Transformer with training head
        optimizers (Tuple[Optimizer, Scheduler], optional): Tuple of Pytorch optimizer and lr scheduler.
        train_dataset (Dataset, optional): Training dataset. Defaults to None.
        eval_dataset (Dataset, optional): Eval/Validation dataset. Defaults to None.
        embedding_model (EmbeddingModel, optional): Embedding model. Used for recovering states during
            state evaluation of the model. Defaults to None.
        viz (Viz, optional): Visualization class. Defaults to None.
    """
    def __init__(self,
        model: TransformerTrain,
        optimizers: Tuple[Optimizer, Scheduler],
        training_loader : DataLoader = None,
        validating_loader: DataLoader = None,
    ) -> None:
        
        self.model = model.to(config["training"]["device"])
        self.training_loader = training_loader
        self.validating_loader = validating_loader
        self.optimizers = optimizers
        set_seed(config["model"]["seed"])

    def train(self) -> None:
        """Trains the transformer model
        """
        optimizer = self.optimizers[0]
        lr_scheduler = self.optimizers[1]
        model = self.model

        # Loop over epochs
        
        train_losses = []
        val_losses = []
        training_loader = self.training_loader
        for epoch in range(1, config["training"]["num_epoch"] + 1):
            
            config["training"]["gradient_accumulation_steps"] = min([config["training"]["gradient_accumulation_steps"], len(training_loader)])
            
            loss_total = 0.0
            model.zero_grad()
            # Loop over mini-batched
            for mbidx, inputs in enumerate(training_loader):
                
                loss0, _, _ =  self.training_step(model, inputs)

                loss_total = loss_total + loss0/len(training_loader)

                # Optimize model
                if (mbidx + 1) % config["training"]["gradient_accumulation_steps"] == 0 or mbidx == len(training_loader)-1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])

                    optimizer.step()
                    lr_scheduler.step(epoch + float(mbidx) / len(training_loader))
                    model.zero_grad()
                    
                    self.epoch = epoch + (mbidx + 1.) / len(training_loader)

            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']
                break

            logger.info("Current Learning rate: {:.09f}".format(cur_lr))
            logger.info("Epoch {:d}: Training loss {:.09f}".format(epoch, loss_total))
            train_losses.append(loss_total)
        
            # Evaluate model
            if(epoch % config["validating"]["val_steps"] == 0 or epoch == 1):
                for param_group in optimizer.param_groups:
                    cur_lr = param_group['lr']
                    break
                logger.info("Current Learning rate: {:.09f}".format(cur_lr))
                #logger.info('Evaluating...')
                val_loss0=self.evaluate(epoch=epoch).item()
                val_losses.append(val_loss0)
            #logger.info("Epoch:{}".format(epoch))
            early_stopping(val_loss0, model, epoch=epoch)
        
            if early_stopping.early_stop:
                logger.info("Early Stopping Activated at Epoch {}".format(epoch))
                break
            if epoch == config["training"]["num_epoch"]:

                # Save model checkpoint
                torch.save(model.state_dict(), config["model"]["path"] + "/model/TransformerGPT.pt")
        torch.save(val_losses, config["model"]["path"]+'/val_loss.pt')      
        torch.save(train_losses, config["model"]["path"] + '/train_loss.pt')    
            

    def training_step(
        self, 
        model: TransformerTrain, 
        inputs: Dict[str, Any]
    ) -> Tuple[float, Tensor, Tensor]:
        """Calls a forward pass of the training model and backprops 
        for a single time-step

        Args:
            model (PhysformerTrain): Transformer model with training head, could be 
            inputs (Dict[str, Any]): Dictionary of model inputs for forward pass

        Returns:
            Tuple[float, Tensor, Tensor]: Tuple containing: loss value, hidden states
                of transformer, attention states of the transformer.
        """
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(config["training"]["device"])

        # Training head forward
        outputs = model(**inputs)
        loss = outputs[0] # Loss value is always the first output

        if config["training"]["gradient_accumulation_steps"] > 1:
            loss = loss / config["training"]["gradient_accumulation_steps"]
        
        # Backward
        loss.backward()

        return loss.item(), outputs[1], outputs[2]

    @torch.no_grad()
    def evaluate(
        self, 
        epoch: int = None
    ) -> Dict[str, float]:
        """Run evaluation and return metrics.

        Args:
            epoch (int, optional): Current epoch, used for naming figures. Defaults to None.

        Returns:
            Dict[str, float]: Dictionary of prediction metrics
        """
        
        validating_loader = self.validating_loader

        eval_error = 0
        
        timestep_error = None

        for mbidx, inputs in enumerate(validating_loader):

            if mbidx == 0:
                timestep_error = torch.zeros(inputs['labels'].size(1))            

            pred_error0, timestep_error0, predicted = self.eval_step(self.model, inputs)

            eval_error += pred_error0/len(validating_loader)
            timestep_error += timestep_error0/len(validating_loader)
        
        logger.info('Validating_error: {:.09f}'.format(eval_error))
        
        return eval_error

    @torch.no_grad()
    def eval_step(
        self, 
        model: TransformerTrain, 
        inputs: Dict[str, Any]
    ) -> Tuple[float, Tensor, Tensor]:
        """Calls a eval pass of the training model.

        Args:
            model (PhysformerTrain): Transformer model with training head
            inputs (Dict[str, Any]): Dictionary of model inputs for forward pass

        Returns:
            Tuple[float, Tensor, Tensor]: Tuple containing: prediction error value, 
                time-step error, predicted embeddings.
        """
        model.eval()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(config["training"]["device"])

        # Training head forward
        outputs = model.evaluate(**inputs)
        pred_error = outputs[0] # Loss value is always the first output

        # Compute loss at each time-step
        mseLoss = nn.MSELoss(reduction='none') # Manual summing
        timestep_error = mseLoss(outputs[1], outputs[2]).mean(dim=(0,2)).cpu()

        return pred_error, timestep_error, outputs[1]

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, verbose=False, epoch: int = None, mse_threshold: float = None, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.verbose = verbose
        self.mse_threshold = mse_threshold
        self.early_stop = False
        self.path = path
        self.epoch = epoch
        self.counter = 0
        self.prev_epoch = 0
        
    def __call__(self, val_loss, model, epoch):

        if val_loss <= self.mse_threshold:
            if self.counter == 0:
                self.prev_epoch = epoch
            
            self.counter += 1
            
            if self.prev_epoch == (epoch-1):
                self.prev_epoch = epoch
            
        if val_loss > self.mse_threshold:
            self.counter = 0
             
        logger.info('Counter value {}'.format(self.counter))
        logger.info('prev_epoch {}'.format(self.prev_epoch))
        logger.info('epoch {}'.format(epoch))
        
        if self.counter == config["training"]["es_counter"]:
            self.early_stop = True
            self.save_checkpoint(val_loss, model, epoch)
                
    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info('Saving model ...')
        torch.save(model.state_dict(), self.path)
        
    
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config["training"]["learning_rate"], max_lr=config["training"]["max_lr"],step_size_up=5,mode="exp_range",gamma=0.85, cycle_momentum=False)
early_stopping = EarlyStopping(mse_threshold=config["training"]["MSE_threshold"], verbose=True, path=config["model"]["path"] + "/model/TransformerGPT.pt")

# =============================================================================
# #plotting learning rate
# import torch
# import matplotlib.pyplot as plt
# 
# for i0 in range(100,101,1):
#     for j in range(1,3,1):
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#         scheduler  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=i0, T_mult=j)
#  
#         lrs = []
# 
#         for i in range(100):
#             scheduler.step()
#             lrs.append(
#                  optimizer.param_groups[0]["lr"]
#              )
#         plt.figure(figsize=(10,5))
#         plt.plot(lrs,'o-')
# 
# 
#         #os.chdir(config["model"]["path"])
#         plt.savefig('learning_rate_t0_%i_tmult_%i.png' %(i0, j) )
#         
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0000001, max_lr=0.002,step_size_up=5,mode="exp_range",gamma=0.85, cycle_momentum=False)
# lrs = []
# 
# plt.figure(figsize=(10,5))
# for i in range(100):
#     optimizer.step()
#     lrs.append(optimizer.param_groups[0]["lr"])
# #     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])
#     scheduler.step()
# 
# plt.plot(lrs)
# 
# plt.savefig('learning_rate CyclicLR.png')
# =============================================================================

trainer = Trainer(
        model, 
        (optimizer, scheduler), 
        training_loader = training_loader, 
        validating_loader = validating_loader 
        
        )

start.record()    
trainer.train()
torch.cuda.synchronize()
end.record()
print("Elapsed Time = %f hr" % (((start.elapsed_time(end)) / 3600000)))  


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

torch.no_grad()

output1=[]
inputs0=[]
r2=[]
mse=[]  
for mbidx, inputs in enumerate(testing_loader):
    model.eval()
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(config["training"]["device"])
            
    outputs = model.evaluate(**inputs)
    output1.append(outputs[1])
    inputs0.append(inputs[k])
    
output1 = torch.cat(output1,dim=0)
inputs0= torch.cat(inputs0,dim=0)

output1=output1.view(output1.size(0),output1.size(1), 8, 196)
inputs0=inputs0.view(inputs0.size(0),inputs0.size(1), 8, 196)

output1=output1[:,:,:3]
inputs0=inputs0[:,:,:3]

output2 = torch.flatten(output1)
inputs2 = torch.flatten(inputs0)

R2=r2_score(output2.cpu().detach(), inputs2.cpu().detach())
r2.append(R2)
MSE=mean_squared_error(output2.cpu().detach(), inputs2.cpu().detach(), squared=True)
mse.append(MSE)
print("r2_score: %.5f" % R2)
print("MSE: %.5f" % MSE)

val_losses=torch.load(config["model"]["path"]+'/val_loss.pt')
train_losses=torch.load(config["model"]["path"]+'/train_loss.pt')

Epoch=torch.arange(1,len(val_losses) + 1 )
fig3 = plt.figure(figsize=(10,5))
plt.title("n_layers: AAAAA, n_heads: BBBBB, block: CCCCC, stride: DDDDD, Base_LR:LLLLL, Max_LR:MMMMM, Batch_size: FFFFF, kfold: HHHHH")
plt.plot(Epoch, val_losses,label="validation")
plt.plot(Epoch, train_losses,label="train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
down, up =plt.gca().get_ylim()
plt.text(Epoch[-1]/2, (up+down)/2, 'MSE:{:.6f} \n R^2:{:.6f} \n Elapsed_time:{:.6f} hr'.format(MSE,R2,((start.elapsed_time(end)) / 3600000)))
plt.legend()
#plt.show()
os.chdir(config["model"]["path_result"])
plt.savefig('learning_curve_%i.png' %config['training']['Job_number']) 
output1[:,0,1,:]=-0.3
inputs0[:,0,1,:]=-0.3

for k in range(0, inputs0.size(0), 10):
    sim_num=k
    X_draw=inputs0[k]
    X_draw=X_draw.cpu().detach().numpy()
    X_draw=np.swapaxes(X_draw,0,2) #((196, 3, 50))
    X_draw=np.swapaxes(X_draw,1,2) #((196, 50, 3))

    group1=X_draw[X_draw[:,0,0]>0.1] #(60, 50, 3)
    group2=X_draw[X_draw[:,0,0]<-0.1] #(72, 50, 3)
    group3=X_draw[(X_draw[:,0,0]<0.1) & (X_draw[:,0,0]>-0.1)] #(64, 50, 3)

    #thin out
    X_cut1 = np.delete(group1, np.s_[0:196:2], 0)
    X_cut1 = np.delete(X_cut1, np.s_[0:196:2], 0)
    X_cut2 = np.delete(group2, np.s_[0:196:2], 0)
    X_cut2 = np.delete(X_cut2, np.s_[0:196:2], 0)
    X_cut3 = np.delete(group3, np.s_[0:196:2], 0)
    X_cut3 = np.delete(X_cut3, np.s_[0:196:2], 0)
    
    def data_for_cylinder_main(center_y,center_z,radius,height_x):
           x = np.linspace(0, height_x, 50)-0.64025
           theta = np.linspace(0, 2*np.pi, 50)
           theta_grid, x_grid=np.meshgrid(theta, x)
           y_grid = radius*np.cos(theta_grid) + center_y
           z_grid = radius*np.sin(theta_grid) + center_z
           return x_grid,y_grid,z_grid
    Xm,Ym,Zm = data_for_cylinder_main(0,0,0.269,1.8805)
    def data_for_cylinder_inlet(center_x,center_z,radius,height_y):
           y = np.linspace(0, -height_y, 50)-0.269
           theta = np.linspace(0, 2*np.pi, 50)
           theta_grid, y_grid=np.meshgrid(theta, y)
           x_grid = radius*np.cos(theta_grid) + center_x
           z_grid = radius*np.sin(theta_grid) + center_z
           return x_grid,y_grid,z_grid
    Xi1,Yi1,Zi1 = data_for_cylinder_inlet(-0.5435,0,0.01875,0.269)
    Xi2,Yi2,Zi2 = data_for_cylinder_inlet(0.0,0,0.01875,0.269)
    Xi3,Yi3,Zi3 = data_for_cylinder_inlet(0.5435,0,0.01875,0.269)
    
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    ax1=fig.add_subplot(111,projection='3d')
    ax1.plot_surface(Xm,Zm,Ym,alpha=0.1,color='blue')
    ax1.plot_surface(Xi1,Zi1,Yi1,alpha=0.1,color='blue')
    ax1.plot_surface(Xi2,Zi2,Yi2,alpha=0.1,color='blue')
    ax1.plot_surface(Xi3,Zi3,Yi3,alpha=0.1,color='blue')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.zaxis.set_major_locator(plt.MaxNLocator(5))
    
    ax1.set_xlabel('X-coordinate (m)', fontsize=12)
    ax1.set_ylabel('Y-coordinate (m)', fontsize=12)
    ax1.set_zlabel('Z-coordinate (m)', fontsize=12)
    
    for i in range(len(X_cut1)):
        X = X_cut1[i][:,0]
        Y = X_cut1[i][:,1]
        Z = X_cut1[i][:,2]
        X[X>1.20]=np.nan
    
        ax1.plot(X, Z, Y, color='magenta', linestyle='-',linewidth=1, markersize=2.5)
    
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax1.plot([xb], [yb], [zb], 'w')
    
    for i in range(len(X_cut2)):
        X = X_cut2[i][:,0]
        Y = X_cut2[i][:,1]
        Z = X_cut2[i][:,2]
        X[X>1.20]=np.nan
    
    
        ax1.plot(X, Z, Y, color='cyan', linestyle='-',linewidth=1, markersize=2.5)
    
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax1.plot([xb], [yb], [zb], 'w')
    
    
    for i in range(len(X_cut3)):
        X = X_cut3[i][:,0]
        Y = X_cut3[i][:,1]
        Z = X_cut3[i][:,2]
        X[X>1.20]=np.nan
    
        ax1.plot(X, Z, Y, color='orange', linestyle='-',linewidth=1, markersize=2.5)
    
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax1.plot([xb], [yb], [zb], 'w')
    
    # plt.title("Particle trajectory (RNN+CNN) on %ith dataset"%sim_num, fontsize=15)
    

    
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    plt.axis('off')

    # Hide axes ticks
    ax1.axes.set_xlim3d(left=-0.8, right=0.8)
    ax1.axes.set_ylim3d(bottom=-0.8, top=0.8)
    ax1.axes.set_zlim3d(bottom=-0.8, top=0.8)
    plt.tight_layout()
    plt.show()
    os.chdir(config["model"]["path"] + "/plot/")
    plt.savefig('plot_%i_original.png'%sim_num)
    
for k in range(0,output1.size(0),10):
    sim_num=k

    y_draw=output1[k]
    y_draw=y_draw.cpu().detach().numpy()

    y_draw=np.swapaxes(y_draw,0,2) #((196, 3, 50))
    y_draw=np.swapaxes(y_draw,1,2) #((196, 50, 3))
    
    group1=y_draw[y_draw[:,0,0]>0.1] #(60, 50, 3)
    group2=y_draw[y_draw[:,0,0]<-0.1] #(72, 50, 3)
    group3=y_draw[(y_draw[:,0,0]<0.1) & (y_draw[:,0,0]>-0.1)] #(64, 50, 3)

    #thin out
    y_cut1 = np.delete(group1, np.s_[0:196:2], 0)
    y_cut1 = np.delete(y_cut1, np.s_[0:196:2], 0)
    y_cut2 = np.delete(group2, np.s_[0:196:2], 0)
    y_cut2 = np.delete(y_cut2, np.s_[0:196:2], 0)
    y_cut3 = np.delete(group3, np.s_[0:196:2], 0)
    y_cut3 = np.delete(y_cut3, np.s_[0:196:2], 0)
    
    #predicted
    fig2 = plt.figure()
    fig2.set_size_inches(18.5, 10.5)
    ax=fig2.add_subplot(111,projection='3d')
    ax.plot_surface(Xm,Zm,Ym,alpha=0.1,color='blue')
    ax.plot_surface(Xi1,Zi1,Yi1,alpha=0.1,color='blue')
    ax.plot_surface(Xi2,Zi2,Yi2,alpha=0.1,color='blue')
    ax.plot_surface(Xi3,Zi3,Yi3,alpha=0.1,color='blue')
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    
    ax.set_xlabel('X-coordinate (m)', fontsize=12)
    ax.set_ylabel('Y-coordinate (m)', fontsize=12)
    ax.set_zlabel('Z-coordinate (m)', fontsize=12)
    
    for i in range(len(y_cut1)):
        X = y_cut1[i][:,0]
        Y = y_cut1[i][:,1]
        Z = y_cut1[i][:,2]
        X[X>1.20]=np.nan
    
    
        ax.plot(X, Z, Y, color='magenta', linestyle='-',linewidth=1, markersize=2.5)
    
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
    
    for i in range(len(y_cut2)):
        X = y_cut2[i][:,0]
        Y = y_cut2[i][:,1]
        Z = y_cut2[i][:,2]
        X[X>1.20]=np.nan
    
        ax.plot(X, Z, Y, color='cyan', linestyle='-',linewidth=1, markersize=2.5)
    
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
    
    
    for i in range(len(y_cut3)):
        X = y_cut3[i][:,0]
        Y = y_cut3[i][:,1]
        Z = y_cut3[i][:,2]
        X[X>1.20]=np.nan
    
        ax.plot(X, Z, Y, color='orange', linestyle='-',linewidth=1, markersize=2.5)
    
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
    
    # plt.title("Particle trajectory (RNN+CNN) on %ith dataset"%sim_num, fontsize=15)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')

    # Hide axes ticks
    ax.axes.set_xlim3d(left=-0.8, right=0.8)
    ax.axes.set_ylim3d(bottom=-0.8, top=0.8)
    ax.axes.set_zlim3d(bottom=-0.8, top=0.8)
    plt.tight_layout()
    plt.show()
    os.chdir(config["model"]["path"] + "/plot/")
    plt.savefig('plot_%i_predicted.png'%sim_num)  
