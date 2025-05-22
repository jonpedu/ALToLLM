import os
import time
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange
from omegaconf import OmegaConf

from net.alto_infer import ALTo
from internvl.train.constants import COODBOOK_SIZE, NUM_HIMT_TOKENS, SEG_START_TOKEN, SEG_END_TOKEN, SEG_TOKEN_TEMPLATE

logger = logging.getLogger(__name__)

class MaskDecoder(ALTo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.codebook_size = COODBOOK_SIZE
        self.num_himt_tokens = NUM_HIMT_TOKENS
        self.num_token_trained = NUM_HIMT_TOKENS

        self.tt_start = 1024
        self.tt_end = self.tt_start + 1
        self.tt_index_start = self.tt_start - 1024
        self.tune_decoder = False

    def init_tt_ids(self, tokenizer):
        self.tt_start = tokenizer.encode(SEG_START_TOKEN)[-1]
        self.tt_end = tokenizer.encode(SEG_END_TOKEN)[-1]
        self.tt_index_start = tokenizer.encode(SEG_TOKEN_TEMPLATE.format(0))[-1]
    
    def count_learnable_params(self):
        count = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                count += param.numel()
        return count
    
    def set_requires_grad(self,requires_grad=False):
        if self.decoder is not None:
            self.decoder.requires_grad_(requires_grad)
            self.pixel_decoder.requires_grad_(requires_grad)
            self.tune_decoder = True

    def set_encoder_requires_grad(self,requires_grad=False):
        if self.encoder is not None:
            self.encoder.requires_grad_(requires_grad)
            self.latent_tokens.requires_grad=requires_grad
            self.quantize.requires_grad_(requires_grad)
            self.sam.requires_grad_(requires_grad)

    @classmethod
    def init_model_from_config(cls, model_path, config_path,
                                device=None, dtype=None,
                                need_encoder=False,
                                need_decoder=False,
                                llm_hidden_size=1024):
        config = OmegaConf.load(config_path)
        config.llm_hidden_size = llm_hidden_size
        model = cls(config)
        if not need_encoder:
            model.encoder = None # remove encoder module in model
        if not need_decoder:
            model.decoder = None # remove decoder module in model
            model.pixel_decoder = None
        
        model.dtype = dtype 
        model.load_weights_from_ckpt(model_path)

        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype=dtype)
        logger.info(f"init {__class__.__name__} with {model.count_learnable_params():,} learnable parameters,dtype={model.dtype}")
        return model

    def load_weights_from_ckpt(self, model_path):
        if not model_path:
            return

        model_info = torch.load(model_path, map_location="cpu")
        if 'model' in model_info:    
            model_weight = model_info['model']
        else:
            model_weight = model_info
        model_weight = {k.replace('module.', ''): v for k, v in model_weight.items()}
        
        self.load_state_dict(model_weight, strict=False)


    def prepare_image(self, image, target_image_size=256):
        # Convert uint8 mask to [0,1] range
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        
        # Handle single channel input (BxHxW -> Bx3xHxW)
        if len(image.shape) == 3:
            image = image.unsqueeze(1)
        if image.shape[1] == 1:
            image = image.expand(-1, 3, -1, -1) 
        
        B, C, H, W = image.shape
        
        # Make square by padding
        max_image_size = max(H, W)
        pad_h = max_image_size - H
        pad_w = max_image_size - W
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Resize to 256x256 if needed
        if max_image_size != target_image_size:
            image = F.interpolate(image, size=(target_image_size, target_image_size), mode='bilinear', align_corners=False)
        
        # Convert to model dtype
        image = image.to(self.dtype)
        return image

    @torch.no_grad()
    def encode_mask(self, image):
        image = self.prepare_image(image)
        z_quantized, result_dict = self.encode(image)
        encoded_tokens = result_dict["min_encoding_indices"]
        length_indices = result_dict["length_indices"]
        encoded_tokens = encoded_tokens.view(encoded_tokens.shape[0], -1)
        return encoded_tokens, length_indices

    def replace_titok_tokens_adaptive(self, input_ids, labels, target_masks):
        
        tt_ids, length_indices = self.encode_mask(target_masks)
        tt_ids = tt_ids.to(input_ids.device) + self.tt_index_start
        
        batch_size = input_ids.size(0)
        new_input_ids = []     
        new_labels = []
        lengths = []
        for i in range(batch_size):
            # 找到当前样本中 tt_placeholder 的位置
            start_position = (input_ids[i] == self.tt_start).nonzero()
            
            if len(start_position) == 0:
                # 如果没有找到 tt_start，直接使用原始序列
                new_input_ids.append(input_ids[i])
                new_labels.append(labels[i])
                lengths.append(32)
                continue
            
            # 获取 tt_start 的位置索引
            start_idx = start_position[0].item()
            first_tt_token = input_ids[i, start_idx+1]
            if first_tt_token == self.tt_index_start:
                length = 32
            elif first_tt_token == self.tt_index_start+1:
                length = int(length_indices[i])
            else:
                raise ValueError(f"Invalid first tt token: {first_tt_token}")
                new_input_ids.append(input_ids[i])
                new_labels.append(labels[i])
                lengths.append(32)
                continue
            lengths.append(length)
            # 构建新序列：前半部分 + mask tokens + 后半部分
            prefix = input_ids[i, :start_idx+1]
            suffix = input_ids[i, start_idx + 33:]  # 跳过34个token (start + 32 + end)

            new_input_ids.append(torch.cat([prefix, tt_ids[i][:length], suffix, torch.full_like(tt_ids[i][length:], self.padding_token)]))


            #label 替换时，用户输入的token不替换
            prefix_label = labels[i, :start_idx+1]
            suffix_label = labels[i, start_idx + 33:]

            if labels[i, start_idx] == -100:
                new_labels.append(labels[i])
            else:
                new_labels.append(torch.cat([prefix_label, tt_ids[i][:length], suffix_label, torch.full_like(tt_ids[i][length:], -100)]))
        
        input_ids = torch.stack(new_input_ids)
        labels = torch.stack(new_labels)
        lengths = torch.tensor(lengths, device=input_ids.device,dtype=torch.long)
        return input_ids, labels, lengths

    def get_train_tt_probs(self, logits, labels):
        batch_size, seq_length, vocab_size = logits.shape

        all_probs = torch.zeros(batch_size, self.num_token_trained, self.codebook_size, device=logits.device)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)

        valid_range_mask = torch.full((self.num_token_trained, vocab_size), float('-inf'), device=logits.device)
        valid_range_mask[:, self.tt_index_start:self.tt_index_start + self.codebook_size] = 0

        labels = labels.view(batch_size, -1)

        mask_indices = torch.zeros(batch_size, self.num_token_trained, dtype=torch.long, device=logits.device)
        for i in range(batch_size):
            start_positions = (labels[i] == self.tt_start).nonzero(as_tuple=True)[0]
            if len(start_positions) == 0 or start_positions[0].item() + self.num_token_trained + 1 >= seq_length:
                continue

            start_idx = start_positions[0].item()
            mask_indices[i] = torch.arange(start_idx + 1, start_idx + 1 + self.num_token_trained)

            valid_mask[i] = True

        expanded_indices = mask_indices.unsqueeze(-1).expand(-1, -1, vocab_size)
        selected_logits = torch.gather(logits, 1, expanded_indices)

        masked_logits = selected_logits + valid_range_mask.unsqueeze(0)

        token_probs = F.softmax(masked_logits * 2, dim=-1)
        titok_token_probs = token_probs[:, :, self.tt_index_start:self.tt_index_start + self.codebook_size]

        valid_mask_clone = valid_mask.clone() # without this, an error will occur as follows:
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
        all_probs[valid_mask_clone] = titok_token_probs[valid_mask_clone]

        return all_probs, valid_mask
    
    @autocast(enabled=True, dtype=torch.bfloat16)
    def decode_prob(self, prob, image_src=None,image_embedding=None,use_norm=True): 
        prob = prob.to(self.dtype)
        codebook = self.quantize.get_codebook_weight().to(self.dtype)    # V x D
        if use_norm:
            prob = torch.nn.functional.normalize(prob, dim=-1)
            codebook = torch.nn.functional.normalize(codebook, dim=-1)
        z = prob @ codebook  # B x T x V * V x D -> B x T x D
        # B x T x D -> B x D x T x 1
        z = rearrange(z, 'b t d -> b d 1 t')
        
        decoded_image, extra_result_dict = self.decode_token_by_vae(z, image_src,image_embedding)
        return decoded_image

    
    def forward(self, x, image_src=None):
        return self.decode_prob(x, image_src)

    def compute_mask_loss(self, logits, labels, target_masks, image_src=None, dice_loss_weight=0.25, cos2fine=0,lengths=None):
        batch_size = logits.shape[0]
        all_probs, valid_mask = self.get_train_tt_probs(logits, labels)
        # Create a mask based on token lengths
        mask = torch.arange(all_probs.shape[1], device=all_probs.device).unsqueeze(0) < lengths.unsqueeze(1)
        # Apply mask to probabilities
        all_probs = all_probs * mask.unsqueeze(-1)

        if valid_mask.any():
            mask_loss_weight = 1
        else:
            mask_loss_weight = 0
            valid_mask[0] = True

        valid_pred_masks = self.decode_prob(all_probs[valid_mask], image_src[valid_mask]).mean(dim=1, keepdim=False)

        valid_target_masks = target_masks[valid_mask]

        mask_bce_loss = compute_bce_loss_prob(valid_pred_masks.float(), valid_target_masks.float())
        mask_dice_loss = compute_dice_loss_prob(valid_pred_masks.float(), valid_target_masks.float())
        
        return mask_loss_weight * (mask_bce_loss + dice_loss_weight * mask_dice_loss)

def compute_bce_loss_prob(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    preds = preds.clamp(min=eps, max=1-eps)
    
    bce_loss = -(targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))
    
    return bce_loss.mean()

def compute_dice_loss_prob(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    preds = preds.clamp(min=eps, max=1-eps)
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + eps) / (
        preds.sum() + targets.sum() + eps
    )
    return 1 - dice