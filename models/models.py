from torch import Tensor
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
# from utils import create_mask

import torchvision
from torch.nn.utils.rnn import pad_sequence

#import utils as utils
from torch.cuda.amp import autocast

""" PyTorch MBART model."""
from transformers import MBartForConditionalGeneration, MBartPreTrainedModel, MBartModel, MBartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.models.mbart.modeling_mbart import shift_tokens_right

from transformers.models.mbart.modeling_mbart import MBartLearnedPositionalEmbedding, MBartEncoderLayer, _expand_mask

from collections import OrderedDict


import copy
import math
import random
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# global definition
from definition import *

from pathlib import Path
from transformers import AutoConfig
import matplotlib.pyplot as plt
import seaborn as sns
PLOT_COUNTER = 0

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
   #     pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(1), :])

def make_resnet(name='resnet18'):
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception('There are no supported resnet model {}.'.format(_('resnet')))

    inchannel = model.fc.in_features
    model.fc = nn.Identity()
    return model

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.resnet = make_resnet(name='resnet18')

    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        x = pad_sequence(x_batch,padding_value=PAD_IDX,batch_first=True)
        return x

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ["P2", "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.temporal_conv(x.permute(0,2,1))
        return x.permute(0,2,1)

def make_head(inplanes, planes, head_type):
    if head_type == 'linear':
        return nn.Linear(inplanes, planes, bias=False)
    else:
        return nn.Identity()

class TextCLIP(nn.Module):
    def __init__(self, config=None, inplanes=1024, planes=1024, head_type='identy'):
        super(TextCLIP, self).__init__()

        self.model_txt = MBartForConditionalGeneration.from_pretrained(config['model']['transformer']).get_encoder()

        self.lm_head = make_head(inplanes, planes, head_type)

    def forward(self, tgt_input):
        txt_logits = self.model_txt(input_ids=tgt_input['input_ids'].cuda(), attention_mask=tgt_input['attention_mask'].cuda())[0]
        output = txt_logits[torch.arange(txt_logits.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        return self.lm_head(output), txt_logits

class ImageCLIP(nn.Module):
    def __init__(self, config, inplanes=1024, planes=1024, head_type='linear') :
        super(ImageCLIP, self).__init__()
        self.config = config
        self.model =  FeatureExtracter()

    def forward(self, src_input):

        x, mid_features = self.model(src_input['input_ids'].cuda(), src_input['src_length_batch']) # [b, n, c]
        return x, mid_features

class Text_Decoder(nn.Module):
    def __init__(self, config):
        super(Text_Decoder, self).__init__()
        self.text_decoder = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_decoder()
        self.lm_head = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_output_embeddings()
        self.register_buffer("final_logits_bias", torch.zeros((1, MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).model.shared.num_embeddings)))


    def forward(self, tgt_input, masked_tgt_input, model_txt):
        with torch.no_grad():
            _, encoder_hidden_states = model_txt(masked_tgt_input)

        decoder_input_ids = shift_tokens_right(tgt_input['input_ids'].cuda(), self.text_decoder.config.pad_token_id)
        decoder_out = self.text_decoder(
                    input_ids = decoder_input_ids,
                    attention_mask = tgt_input['attention_mask'].cuda(),
                    encoder_hidden_states = encoder_hidden_states,
                    encoder_attention_mask = masked_tgt_input['attention_mask'].cuda(),
                    return_dict = True,
                    )
        lm_logits = self.lm_head(decoder_out[0]) + self.final_logits_bias

        return lm_logits

class FeatureExtracter(nn.Module):
    def __init__(self, frozen=False):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = resnet()
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)

        if frozen:
            for param in self.conv_2d.parameters():
                param.requires_grad = False
            for param in self.conv_1d.parameters():
                param.requires_grad = False

    def forward(self,
                src: Tensor,
                src_length_batch
                ):
        mid_features = self.conv_2d(src,src_length_batch)
        src = self.conv_1d(mid_features)

        return src, mid_features

class VisionEncoder(nn.Module):
    """
    Mbart encoder for VLP stage
    """
    def __init__(self, config, inplanes=1024, planes=1024, head_type='linear') :
        super(VisionEncoder, self).__init__()
        self.config = config

        if "config_file" in config['model'].keys():
            configuration = MBartConfig.from_pretrained(config['model']['config_file'])
            self.trans_encoder  = MBartForConditionalGeneration._from_config(config=configuration).get_encoder()
        else:
            self.trans_encoder = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_encoder()
        self.cls_token = nn.Parameter(torch.randn(1, 1, inplanes))

        self.lm_head = make_head(inplanes, planes, head_type)

    def forward(self, attention_mask, x):
        B, N, C = x.shape
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=B)
        x = torch.cat((cls_token, x), dim=1) # [b, t, dim=1024]
        attention_mask = F.pad(attention_mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]

        # Because of different temporal encoding, modalities may have different (more) T frames than image_features. Therefore, reduce them
        if x.shape[1] != attention_mask.shape[-1]:
            x = x[:, :attention_mask.shape[-1], :]

        outs = self.trans_encoder(inputs_embeds=x, attention_mask=attention_mask.cuda(), return_dict=True)
        last_hidden_state = outs['last_hidden_state']
        output = self.lm_head(last_hidden_state[:, 0, :])
        return output, last_hidden_state, attention_mask

class CiCo:
    def cross_lingual_similarity(self, visual_output, sequence_output, feature1_att_mask, feature2_att_mask, logit_scale):
        # Reference for this function: https://github.com/FangyunWei/SLRT/tree/main/CiCo (CLCL/modules/modeling.py)

        # visual_output: (B, T, 1024) - sequence_output: (B, L, 1024)
        # normalize
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        # matrix multiplication
        dot_p = torch.einsum("ais,bjs->abij", [visual_output, sequence_output])  # (B, B, T, L)
        # row-wise softmax and element-wise multiply, then row-wise sum
        after_softmax_i2t = torch.nansum(dot_p * torch.softmax(dot_p / 0.07, dim=-1), dim=-1)  # (B, B, T)
        # col-wise softmax and element-wise multiply, then col-wise sum
        after_softmax_t2i = torch.nansum(dot_p * torch.softmax(dot_p / 0.07, dim=-2), dim=-2)  # (B, B, L)

        # extend attention masks
        feature1_att_mask = feature1_att_mask.unsqueeze(1).repeat(1, visual_output.shape[0], 1).to(dot_p.device)  # (B, B, T)
        feature2_att_mask = feature2_att_mask.unsqueeze(0).repeat(sequence_output.shape[0], 1, 1).to(dot_p.device)  # (B, B, L)
        after_softmax_i2t[~feature1_att_mask] = 0
        after_softmax_t2i[~feature2_att_mask] = 0

        # average and obtain similarity matrix
        I2T_sim = logit_scale * torch.nansum(after_softmax_i2t, dim=-1) / feature1_att_mask.sum(dim=-1)  # (B, B)
        T2I_sim = logit_scale * torch.nansum(after_softmax_t2i, dim=-1) / feature2_att_mask.sum(dim=-1)  # (B, B)

        return I2T_sim, T2I_sim

    def cross_lingual_similarity_v2(self, visual_output, sequence_output, attention_mask, tgt_mask, logit_scale):
        # This function is based on C2RL's authors' implementation, which was kindly shared with us for reproducibility.

        # visual_output: (B, T, 1024) - sequence_output: (B, L, 1024)
        # normalize
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        # matrix multiplication
        sim = torch.einsum("ais,bjs->abij", [visual_output, sequence_output])  # (B, B, T, L)

        batch_size_t = sequence_output.shape[0]
        batch_size_v = visual_output.shape[0]
        # row-wise softmax
        att_mask = torch.where(tgt_mask == 0, torch.tensor(float('1e-5')).cuda(), tgt_mask)
        i2t_sim = sim * att_mask.unsqueeze(0).repeat(batch_size_v, 1, 1).unsqueeze(2)
        att = torch.softmax(i2t_sim / 0.07, dim=3)

        after_softmax_i2t = torch.nansum(i2t_sim * att, dim=3)
        visual_attention_mask = (attention_mask == 1)
        video_mask_extend = visual_attention_mask.unsqueeze(1).repeat(1, batch_size_t, 1).cuda()
        after_softmax_i2t[~video_mask_extend] = 0
        I2T_sim = logit_scale * torch.nansum(after_softmax_i2t, dim=-1) / torch.sum(video_mask_extend, dim=-1)

        # col-wise softmax
        att_mask = torch.where(attention_mask == 0, torch.tensor(float('1e-5')).cuda(), attention_mask)
        t2i_sim = sim * att_mask.unsqueeze(1).repeat(1, batch_size_t, 1).unsqueeze(3)
        att = torch.softmax(t2i_sim / 0.07, dim=2)

        after_softmax_t2i = torch.nansum(t2i_sim * att, dim=2)
        tgt_mask = (tgt_mask == 1)
        text_mask_extend2 = tgt_mask.unsqueeze(0).repeat(batch_size_v, 1, 1).cuda()
        after_softmax_t2i[~text_mask_extend2] = 0
        T2I_sim = logit_scale * torch.nansum(after_softmax_t2i * text_mask_extend2, dim=-1) / torch.sum(text_mask_extend2, dim=-1)

        ground_truth = torch.eye(batch_size_v, device=I2T_sim.device, dtype=I2T_sim.dtype, requires_grad=False)

        loss_itc = (torch.nn.functional.cross_entropy(I2T_sim, ground_truth, label_smoothing=0.2)
                    + torch.nn.functional.cross_entropy(T2I_sim.T, ground_truth, label_smoothing=0.2)
                    ) / 2

        return I2T_sim, T2I_sim, loss_itc

class SLRCLIP(nn.Module, CiCo):
    def __init__(self, config, embed_dim=1024, model_type="gfslt"):
        super(SLRCLIP, self).__init__()
        self.model_txt = TextCLIP(config, inplanes=embed_dim, planes=embed_dim)
        self.model_images = ImageCLIP(config, inplanes=embed_dim, planes=embed_dim)

        self.vision_encoder = VisionEncoder(config, inplanes=embed_dim, planes=embed_dim, head_type='linear')

        self.config = config
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.model_type = model_type

    def get_model_txt(self):
        return self.model_txt

    @property
    def get_encoder_hidden_states(self):
        return self.encoder_hidden_states

    def forward(self, src_input, tgt_input):
        # normalized text features
        text_features, self.encoder_hidden_states = self.model_txt(tgt_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features, mid_features = self.model_images(src_input)
        # call transformer encoder
        image_features, all_image_features, attention_mask = self.vision_encoder(attention_mask=src_input["attention_mask"], x=image_features)
        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)


        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = {}
        batch_size = src_input["attention_mask"].shape[0]

        if self.model_type == "gfslt" or self.model_type == "signcl":
            logits["logits_per_image"] = logit_scale * image_features @ text_features.t()
            logits["logits_per_text_image"] = logit_scale * text_features @ image_features.t()

        elif self.model_type == "cico":
            feature1_att_mask = (attention_mask == 1)
            feature2_att_mask = (tgt_input["attention_mask"] == 1)
            I2T_sim, T2I_sim = self.cross_lingual_similarity(all_image_features, self.encoder_hidden_states, feature1_att_mask, feature2_att_mask, logit_scale)
            logits["logits_per_image"] = I2T_sim
            logits["logits_per_text_image"] = T2I_sim

        dtype = logits["logits_per_image"].dtype
        ground_truth = torch.eye(batch_size, device=text_features.device, dtype=dtype, requires_grad=False)

        # mid_features is used in SignCL to calculate additional contrastive loss on adjacent frames
        return logits, ground_truth, mid_features

def config_decoder(config):
    decoder_type = config["model"].get("decoder_type", "LD")

    if decoder_type == 'LD':
        return MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'], ignore_mismatched_sizes = True, config = AutoConfig.from_pretrained(Path(config['model']['visual_encoder'])/'config.json'))
    elif decoder_type == 'LLMD':
        return MBartForConditionalGeneration.from_pretrained(config['model']['transformer'], ignore_mismatched_sizes = True, config = AutoConfig.from_pretrained(Path(config['model']['transformer'])/'LLMD_config.json'))
    else:
        # full 12-layers Mbart
        config_path = Path(config['model']['transformer']) / 'config.json'
        model_config = AutoConfig.from_pretrained(config_path)
        if decoder_type == "mbart_12layers":
            return MBartForConditionalGeneration.from_pretrained(config['model']['transformer'],
                                                             ignore_mismatched_sizes=True,
                                                             config=model_config)
        elif decoder_type == "mbart_12layers_dropout0.3":
            model_config.dropout = 0.3
            return MBartForConditionalGeneration.from_pretrained(config['model']['transformer'],
                                                             ignore_mismatched_sizes=True,
                                                             config=model_config)

class gloss_free_model(nn.Module, CiCo):
    def __init__(self, config, args, embed_dim=1024, pretrain=None):
        super(gloss_free_model, self).__init__()
        self.config = config
        self.args = args
        self.model_type = args.model_type

        self.backbone = FeatureExtracter(frozen=args.frozenFeatureExtractor)
        self.mbart = config_decoder(config)

        if config['model']['sign_proj']:
            self.sign_emb = V_encoder(emb_size=embed_dim, feature_size=embed_dim, config=config)
            self.embed_scale = math.sqrt(embed_dim) if config['training']['scale_embedding'] else 1.0
        else:
            self.sign_emb = nn.Identity()
            self.embed_scale = 1.0

        if self.model_type in ["c2rl"]:
            # for CICO loss
            self.model_txt = TextCLIP(config, inplanes=embed_dim, planes=embed_dim)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def feature_extract_forward(self, src_input):
        attention_mask = src_input['attention_mask']

        img_features, mid_features = self.backbone(src_input['input_ids'].cuda(), src_input['src_length_batch'])
        # img_features.shape = torch.Size([Batch, Seq/4, 1024])
        # mid_features.shape = torch.Size([Batch, Seq, 1024])

        # VL adapter
        inputs_embeds = self.sign_emb(img_features)
        inputs_embeds = self.embed_scale * inputs_embeds

        return inputs_embeds, attention_mask, mid_features

    def forward(self, src_input, tgt_input):

        inputs_embeds, attention_mask, mid_features = self.feature_extract_forward(src_input)

        out = self.mbart(inputs_embeds=inputs_embeds,
                         attention_mask=attention_mask.cuda(),
                         # decoder_input_ids = tgt_input['input_ids'].cuda(),
                         labels=tgt_input['input_ids'].cuda(),
                         decoder_attention_mask=tgt_input['attention_mask'].cuda(),
                         return_dict=True,
                         output_attentions = True
                         )
        #  self.plot_attention_map(out["cross_attentions"],attention_mask )

        if not self.model_type in ["c2rl"]:
            return out['logits'], mid_features # torch.Size([B, T, 2172])
        else:
            # ---- CICO loss  ----
            text_features, text_encoder_last_hidden_state = self.model_txt(tgt_input)
            # text_features = (B, 1024), text_encoder_last_hidden_state = (B, T, 1024)

            image_hidden_states = out.encoder_last_hidden_state

            feature1_att_mask = (attention_mask == 1)
            feature2_att_mask = (tgt_input["attention_mask"] == 1)
            # cosine similarity as logits
            cico_logits = {}
            logit_scale = self.logit_scale.exp()
            if self.args.model_type == "c2rl":
                I2T_sim, T2I_sim, loss_itc = self.cross_lingual_similarity_v2(image_hidden_states,
                                                                              text_encoder_last_hidden_state,
                                                                              attention_mask.cuda(),
                                                                              tgt_input['attention_mask'].cuda(), logit_scale)
                cico_logits["loss_itc"] = loss_itc

            cico_logits["logits_per_image"] = I2T_sim
            cico_logits["logits_per_text_image"] = T2I_sim
            cico_logits_gt = torch.eye(src_input["attention_mask"].shape[0], device=text_features.device,
                                       dtype=cico_logits["logits_per_image"].dtype, requires_grad=False)

            #  self.plot_attention_map(out["cross_attentions"],attention_mask )
            return out['logits'], cico_logits, cico_logits_gt  # torch.Size([B, T, 2172])



    def plot_attention_map(self, cross_attentions, attention_mask):
        global PLOT_COUNTER
        last_layer_attn = cross_attentions[-1]
        attn_map = last_layer_attn.mean(dim=1) # average all heads

        i = 0
        # take the i'th sample, get only valid frames (not padding)
        attn_map = attn_map[i].T
        attn_map = torch.where(attn_map < 0.1, torch.tensor(0.0, device=attn_map.device), attn_map)
        valid_frames = attention_mask[i].cpu().numpy().astype(bool)
        attn_map = attn_map[valid_frames]

        GROUP_BY = 2
        TOKEN_GROUP_BY = 2

        frame_count = int(attn_map.shape[0])
        token_count = int(attn_map.shape[1])
        num_frame_groups = frame_count // GROUP_BY
        num_token_groups = token_count // TOKEN_GROUP_BY

        # group frames
        attn_map_grouped = attn_map[:num_frame_groups * GROUP_BY]
        attn_map_grouped = attn_map_grouped.reshape(num_frame_groups, GROUP_BY, -1)  # (group_count, 5, token_count)
        attn_map_grouped = attn_map_grouped.sum(dim=1)  # (group_count, token_count)

        # group tokens
        attn_map_grouped = attn_map_grouped[:, :num_token_groups * TOKEN_GROUP_BY]
        attn_map_grouped = attn_map_grouped.reshape(num_frame_groups, num_token_groups, TOKEN_GROUP_BY)  # (frame_group_count, token_group_count, 5)
        attn_map_grouped = attn_map_grouped.sum(dim=2)  # (group_count, group_count)


        plt.figure(figsize=(int(token_count/2), int(frame_count/2)))
        sns.heatmap(attn_map_grouped.detach().cpu().numpy(), cmap='gray', cbar=False)

        # plt.figure(figsize=( int(frame_count / 2), int(token_count / 2)))
        # sns.heatmap(attn_map_grouped.T.detach().cpu().numpy(), cmap='gray', cbar=False, vmin=0, vmax=1)


        plt.xticks([])
        plt.yticks([])

        # save_path =  f"/demo/attention_map_{PLOT_COUNTER}.png"
        # PLOT_COUNTER += 1
        # plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.clf()


    def generate(self, src_input, max_new_tokens, num_beams, decoder_start_token_id, num_return_sequences=1, do_sample=False):
        inputs_embeds, attention_mask, mid_features = self.feature_extract_forward(src_input)

        out = self.mbart.generate(inputs_embeds=inputs_embeds,
                                  attention_mask=attention_mask.cuda(), max_new_tokens=max_new_tokens,
                                  num_beams=num_beams,
                                  decoder_start_token_id=decoder_start_token_id,
                                  num_return_sequences=num_return_sequences, do_sample=do_sample
                                  )
        return out

class V_encoder(nn.Module):
    def __init__(self,
                 emb_size,
                 feature_size,
                 config,
                 ):
        super(V_encoder, self).__init__()

        self.config = config
        self.normalization = config["model"].get("sign_proj_normalization", "BN")
        modules = []
        if self.normalization == "BN":
            modules.append(nn.BatchNorm1d(emb_size))
        elif self.normalization == "LN":
            modules.append(nn.LayerNorm(emb_size))
        modules.append(nn.ReLU(inplace=True))

        self.src_emb = nn.Linear(feature_size, emb_size)
        self.bn_ac = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, src,
                ):

        src = self.src_emb(src)
        if self.normalization == "LN":
            src = self.bn_ac(src)
        else:
            src = self.bn_ac(src.permute(0,2,1)).permute(0,2,1)

        return src