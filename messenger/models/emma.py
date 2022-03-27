'''
Implements the EMMA model
'''

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from numpy import sqrt as sqrt
from transformers import AutoModel, AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_random

from messenger.models.utils import nonzero_mean, Encoder


class EMMA(nn.Module):
    def __init__(
            self,
            state_h=10,
            state_w=10,
            action_dim=5,
            hist_len=3,
            n_latent_var=128,
            emb_dim=256,
            f_maps=64,
            kernel_size=2,
            n_hidden_layers=1,
            forward_type='original',
            device=None):

        super().__init__()

        # calculate dimensions after flattening the conv layer output
        lin_dim = f_maps * (state_h - (kernel_size - 1)) * (
            state_w - (kernel_size - 1))
        self.conv = nn.Conv2d(
            hist_len * 256,
            f_maps,
            kernel_size)  # conv layer

        self.state_h = state_h
        self.state_w = state_w
        self.action_dim = action_dim
        self.emb_dim = emb_dim
        self.attn_scale = sqrt(emb_dim)

        self.sprite_emb = nn.Embedding(
            25, emb_dim, padding_idx=0)  # sprite embedding layer

        hidden_layers = (
            nn.Linear(
                n_latent_var,
                n_latent_var),
            nn.LeakyReLU()) * n_hidden_layers
        self.action_layer = nn.Sequential(
            nn.Linear(lin_dim, n_latent_var),
            nn.LeakyReLU(),
            *hidden_layers,
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(lin_dim, n_latent_var),
            nn.LeakyReLU(),
            *hidden_layers,
            nn.Linear(n_latent_var, 1)
        )

        # key value transforms
        self.txt_key = nn.Linear(768, emb_dim)
        self.scale_key = nn.Sequential(
            nn.Linear(768, 1),
            nn.Softmax(dim=-2)
        )

        self.txt_val = nn.Linear(768, emb_dim)
        self.scale_val = nn.Sequential(
            nn.Linear(768, 1),
            nn.Softmax(dim=-2)
        )

        if device:
            self.device = device
        else:
            self.device = torch.device("cpu")
        self.forward_type = forward_type

        # get the text encoder
        text_model = self._load_transformer_model('bert-base-uncased')
        tokenizer = self._load_transformer_tokenizer('bert-base-uncased')
        self.encoder = Encoder(
            model=text_model,
            tokenizer=tokenizer,
            device=self.device)
        self.to(device)

    @retry(stop=stop_after_attempt(10), wait=wait_random(5, 30))
    def _load_transformer_model(self, name: str):
        return AutoModel.from_pretrained(name)

    @retry(stop=stop_after_attempt(10), wait=wait_random(5, 30))
    def _load_transformer_tokenizer(self, name: str):
        return AutoTokenizer.from_pretrained(name)

    def to(self, device):
        '''
        Override the .to() method so that we can store the device as an attribute
        and also update the device for self.ncoder (which does not inherit nn.Module)
        '''
        self.device = device
        self.encoder.to(device)
        return super().to(device)

    def attention(self, query, key, value):
        '''
        Cell by cell attention mechanism. Uses the sprite embeddings as query. Key is
        text embeddings
        '''
        kq = query @ key.transpose(1, 2).unsqueeze(1).unsqueeze(1)
        mask = (kq != 0)  # keep zeroed-out entries zero
        kq = kq / self.attn_scale  # scale to prevent vanishing grads
        weights = F.softmax(kq, dim=-1) * mask
        weighted_vals = weights.unsqueeze(-1) * value.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        return torch.mean(weighted_vals, dim=-2), weights

    # def forward(self, obs, manual):
    def forward(self, obs, manual=None, entity=None, avatar=None):
        
        if self.forward_type == 'original':
            # The original forward does not expect batch dimension
            if manual is None:
                raise ValueError()
            if entity is not None:
                raise ValueError()
            if avatar is not None:
                raise ValueError()

            entity_obs = obs["entities"]
            avatar_obs = obs["avatar"]
            temb = self.encoder.encode(manual)

            # add batch dimension
            entity_obs = entity_obs.unsqueeze(0)
            avatar_obs = avatar_obs.unsqueeze(0)
            temb = temb.unsqueeze(0)
        elif self.forward_type == 'pfrl':
            manual, entity_obs, avatar_obs = obs
            temb = manual
        else:
            raise ValueError()

        batch_size = entity_obs.shape[0]

        # embedding for the avatar object, which will not attend to text
        avatar_emb = nonzero_mean(self.sprite_emb(avatar_obs))

        # take the non_zero mean of embedded objects, which will act as
        # attention query
        query = nonzero_mean(self.sprite_emb(entity_obs))

        # Attention
        key = self.txt_key(temb)
        key_scale = self.scale_key(temb)  # (num sent, sent_len, 1)
        key = key * key_scale
        key = torch.sum(key, dim=-2)

        value = self.txt_val(temb)
        val_scale = self.scale_val(temb)
        value = value * val_scale
        value = torch.sum(value, dim=-2)

        obs_emb, _ = self.attention(query, key, value)

        # compress the channels from KHWC to HWC' where K is history length
        obs_emb = obs_emb.view(batch_size, self.state_h, self.state_w, -1)
        avatar_emb = avatar_emb.view(batch_size, self.state_h, self.state_w, -1)
        obs_emb = (obs_emb + avatar_emb) / 2.0

        # permute from HWC to NCHW and do convolution
        obs_emb = obs_emb.permute(0, 3, 1, 2)
        obs_emb = F.leaky_relu(self.conv(obs_emb)).reshape(batch_size, -1)

        action_probs = self.action_layer(obs_emb)
        value = self.value_layer(obs_emb)

        if self.forward_type == 'original':
            actions = torch.argmax(action_probs, dim=1)
            if random.random() < 0.05:  # random actions with 0.05 prob
                actions = [random.randrange(0, self.action_dim)
                           for _ in range(0, batch_size)]
            return actions[0]
        elif self.forward_type == 'pfrl':
            return action_probs, value
        else:
            raise ValueError()
