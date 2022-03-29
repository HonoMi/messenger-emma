'''
Implements the EMMA model
'''

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt as sqrt
from transformers import AutoModel, AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_random
from self_attention_cv import MultiHeadSelfAttention
# from self_attention_cv.pos_embeddings import AbsPosEmb1D, RelPosEmb1D
from self_attention_cv.pos_embeddings import PositionalEncodingSin

from messenger.models.utils import nonzero_mean, Encoder


class AttnVec(nn.Module):

    def __init__(self, input_dim: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Linear(input_dim, emb_dim)
        self.scale = nn.Sequential(nn.Linear(input_dim, 1), nn.Softmax(dim=-2))

    def forward(self, emb: torch.Tensor):
        attn_emb = self.emb(emb)
        scale = self.scale(emb)
        scaled_emb = attn_emb * scale
        scaled_emb = torch.sum(scaled_emb, dim=-2)
        return scaled_emb


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
            do_image_self_attention=False,
            n_image_self_attention_heads=4,
            do_text_attention_after_conv=False,
            do_contrastive_learning=False,
            tf_writer=None,
            device='cuda',
    ):

        super().__init__()
        bert_emb_dim = 768

        # calculate dimensions after flattening the conv layer output
        lin_dim = f_maps * (state_h - (kernel_size - 1)) * (
            state_w - (kernel_size - 1))
        self.conv = nn.Conv2d(
            hist_len * 256,
            f_maps,
            kernel_size)  # conv layer

        self.do_image_self_attn = do_image_self_attention
        if self.do_image_self_attn:
            self.image_self_attn = MultiHeadSelfAttention(dim=f_maps, heads=n_image_self_attention_heads)

            n_image_tokens = (state_h - 1) * (state_w - 1)
            self.image_pos_emb = PositionalEncodingSin(dim=f_maps, max_tokens=n_image_tokens)
            # The output tensor shape of the following classes are somewhat unexpected.
            # self.image_pos_emb = RelPosEmb1D(tokens = n_image_tokens,
            #                                  dim_head=f_maps,
            #                                  heads=1)
            # self.image_pos_emb = AbsPosEmb1D(tokens = n_image_tokens,
            #                                  dim_head=f_maps)
        else:
            self.image_self_attn = None
            self.image_pos_emb = None

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

        def build_attn_linear(input_dim: int, output_dim: int):
            return nn.Linear(input_dim, output_dim)

        def build_attn_scale(input_dim: int):
            return nn.Sequential(nn.Linear(input_dim, 1), nn.Softmax(dim=-2))

        # key value transforms
        self.txt_attn_key = AttnVec(bert_emb_dim, emb_dim)
        self.txt_attn_val = AttnVec(bert_emb_dim, emb_dim)

        self.do_text_attention_after_conv = do_text_attention_after_conv
        if self.do_text_attention_after_conv:
            self.txt_attn_after_conv_key = AttnVec(bert_emb_dim, f_maps)
            self.txt_attn_after_conv_val = AttnVec(bert_emb_dim, f_maps)
        else:
            self.txt_attn_after_conv_key = None
            self.txt_attn_after_conv_val = None

        # if device:
        #     self.device = device
        # else:
        #     self.device = torch.device("cpu")
        self.forward_type = forward_type

        # get the text encoder
        text_model = nn.DataParallel(self._load_transformer_model('bert-base-uncased'))
        tokenizer = self._load_transformer_tokenizer('bert-base-uncased')
        self.encoder = Encoder(
            model=text_model,
            tokenizer=tokenizer,
            # device=self.device
            device=device,
        )
        # self.to(device)

        self.do_contrastive_learning = do_contrastive_learning
        if self.do_contrastive_learning:
            self.contrastive_layer = nn.Sequential(
                nn.Linear(lin_dim, n_latent_var),
                nn.LeakyReLU(),
                *hidden_layers,
                nn.Linear(n_latent_var, 2)
            )
            self.contrastive_criteria = nn.CrossEntropyLoss()
        else:
            self.contrastive_layer = None
            self.contrastive_criteria = None

        self._n_forward_called = 0
        self._tf_writer = tf_writer


    @retry(stop=stop_after_attempt(10), wait=wait_random(5, 30))
    def _load_transformer_model(self, name: str):
        return AutoModel.from_pretrained(name)

    @retry(stop=stop_after_attempt(10), wait=wait_random(5, 30))
    def _load_transformer_tokenizer(self, name: str):
        return AutoTokenizer.from_pretrained(name)

    # def to(self, device):
    #     '''
    #     Override the .to() method so that we can store the device as an attribute
    #     and also update the device for self.ncoder (which does not inherit nn.Module)
    #     '''
    #     # self.device = device
    #     self.encoder.to(device)
    #     return super().to(device)

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

    def forward(self, obs, manual=None):

        if self.forward_type == 'original':
            # The original forward does not expect batch dimension
            if manual is None:
                raise ValueError()

            entity_obs = obs["entities"]
            avatar_obs = obs["avatar"]

            if str(self.encoder.device) != str(avatar_obs.device):
                self.encoder = self.encoder.to(avatar_obs.device)
            temb = self.encoder.encode(manual)

            # add batch dimension
            entity_obs = entity_obs.unsqueeze(0)
            avatar_obs = avatar_obs.unsqueeze(0)
            temb = temb.unsqueeze(0)

        elif self.forward_type == 'pfrl':
            if manual is not None:
                raise ValueError()
            manual, entity_obs, avatar_obs = obs
            temb = manual
        else:
            raise ValueError()

        if self.do_contrastive_learning and torch.is_grad_enabled():
            # if str(self.contrastive_criteria.device) != str(entity_obs.device):
            #     self.contrastive_criteria = self.contrastive_criteria.to(entity_obs.device)
            contrastive_criteria = self.contrastive_criteria.to(temb.device)

            negative_temb = temb[torch.randperm(temb.shape[0])]
            _, _, negative_logits = self._forward(entity_obs, avatar_obs, negative_temb)
            negative_labels = torch.zeros(negative_logits.shape[0], dtype=torch.int64).to(negative_logits.device)
            contrastive_loss = contrastive_criteria(negative_logits, negative_labels)

            action_probs, value, positive_logits = self._forward(entity_obs, avatar_obs, temb)
            positive_labels = torch.ones(negative_logits.shape[0], dtype=torch.int64).to(negative_logits.device)
            contrastive_loss += contrastive_criteria(positive_logits, positive_labels)

            contrastive_loss.backward(retain_graph=True)
            if self._tf_writer is not None:
                self._tf_writer.add_scalar('agent/contrastive_loss',
                                           contrastive_loss.cpu().detach().numpy().item(),
                                           self._n_forward_called)
        else:
            action_probs, value, _ = self._forward(entity_obs, avatar_obs, temb)

        self._n_forward_called += 1
        return action_probs, value

    def _forward(self, entity_obs, avatar_obs, temb):

        batch_size = entity_obs.shape[0]

        # embedding for the avatar object, which will not attend to text
        avatar_emb = nonzero_mean(self.sprite_emb(avatar_obs))

        # take the non_zero mean of embedded objects, which will act as
        # attention query
        query = nonzero_mean(self.sprite_emb(entity_obs))

        # Attention
        key = self.txt_attn_key(temb)
        value = self.txt_attn_val(temb)
        obs_emb, _ = self.attention(query, key, value)

        # compress the channels from KHWC to HWC' where K is history length
        obs_emb = obs_emb.view(batch_size, self.state_h, self.state_w, -1)
        avatar_emb = avatar_emb.view(batch_size, self.state_h, self.state_w, -1)
        obs_emb = (obs_emb + avatar_emb) / 2.0

        # permute from HWC to NCHW and do convolution
        obs_emb = obs_emb.permute(0, 3, 1, 2)
        obs_emb = F.leaky_relu(self.conv(obs_emb))

        if self.do_image_self_attn:
            obs_token_embs = obs_emb.view(obs_emb.shape[0], obs_emb.shape[1], -1).permute(0, 2, 1)
            obs_token_embs = self.image_pos_emb(obs_token_embs)
            obs_token_embs = self.image_self_attn(obs_token_embs)
            obs_emb = obs_token_embs.permute(0, 2, 1).reshape(*obs_emb.shape)

        if self.do_text_attention_after_conv:
            key = self.txt_attn_after_conv_key(temb)
            value = self.txt_attn_after_conv_val(temb)
            obs_emb = obs_emb.permute(0, 2, 3, 1).unsqueeze(1)    # channel last add pseudo channel dim
            obs_emb, _ = self.attention(obs_emb, key, value)
            obs_emb.squeeze(1).permute(0, 3, 1, 2)

        obs_emb = obs_emb.reshape(batch_size, -1)
        action_probs = self.action_layer(obs_emb)
        value = self.value_layer(obs_emb)
        if self.do_contrastive_learning:
            contrastive_logits = self.contrastive_layer(obs_emb)
        else:
            contrastive_logits = None

        if self.forward_type == 'original':
            actions = torch.argmax(action_probs, dim=1)
            if random.random() < 0.05:  # random actions with 0.05 prob
                actions = [random.randrange(0, self.action_dim)
                           for _ in range(0, batch_size)]
            return actions[0]
        elif self.forward_type == 'pfrl':
            return action_probs, value, contrastive_logits
        else:
            raise ValueError()
