import torch
import torch.utils.data
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from internvl.train.constants import (
    COODBOOK_SIZE, NUM_HIMT_TOKENS, SEG_START_TOKEN, SEG_END_TOKEN, SEG_TOKEN_TEMPLATE
)
from trl.models import unwrap_model_for_generation

import numpy as np
from PIL import Image

def compute_iou(gt_mask, mask_image):
    """Compute intersection over union (IoU) between two masks."""
    gt_mask = gt_mask > 0.5
    mask_image = mask_image > 0.5
    intersection = np.logical_and(gt_mask, mask_image)
    union = np.logical_or(gt_mask, mask_image)
    intersection = np.sum(intersection)
    union = np.sum(union)
    iou = intersection / (union + 1e-10)
    if union == 0:
        iou = 1.0
    return intersection, union, iou


def update_metrics(mask_images, batch_samples):
    """Update IoU metrics for a batch."""
    ious = []
    for data, mask_image in zip(batch_samples, mask_images):
        gt_mask = data["mask"]
        gt_size = max(gt_mask.shape[0], gt_mask.shape[1])
        mask_image = Image.fromarray(mask_image).resize((gt_size, gt_size), Image.NEAREST)
        mask_image = np.array(mask_image)
        mask_image = mask_image[:gt_mask.shape[0], :gt_mask.shape[1]]
        _, _, iou = compute_iou(gt_mask, mask_image)
        ious.append(iou)
    return ious


def reward_iou(mask_images, target_masks, valid_mask):
    valid_mask_images = torch.zeros_like(mask_images)
    valid_mask_images[valid_mask] = mask_images[valid_mask]

    mask_images = valid_mask_images.detach()
    mask_images = mask_images.float().cpu().numpy()
    target_masks = target_masks.float().cpu().numpy()

    batch_samples = [{'mask': gt_mask} for gt_mask in target_masks]

    ious = update_metrics(mask_images, batch_samples)
    return ious

def ids_are_same(ids1, ids2):
    """ids1 and ids2 are tensors, element of them are token ids (int type)"""
    is_same = ids1 == ids2
    num_false = (is_same == False).sum().item()
    return num_false == 0


class SegGRPOTrainer(Trainer):
    # Refer to GRPOTrainer and Qwen2VLGRPOTrainer from open-r1-multimodal
    # Refer to projs/UUMM/eval_segmentation/eval_simple_wjz.py
    # Only inference is needed. Model needs to be renewed.

    def __init__(
        self,
        model, 
        ref_model,
        tokenizer,
        args,
        train_dataset,
        eval_dataset,
        data_collator,
        **kwargs
    ):
        self.ref_model = ref_model
        self.length_weight = kwargs.pop('length_weight', 0.0)

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            **kwargs
        )

        if self.ref_model is not None:
            self.ref_model.mask_decoder = None
            self.ref_model.vision_model = model.vision_model
            self.ref_model = self.ref_model.to(self.accelerator.device)
            print(f"self.ref_model: {self.ref_model.device}")

        self.kl_beta = 0.001
        self.grpo_group_size = 12
        self.temperature = 1.0
        self.top_k = 10
        self.top_p = None

        self.myprint(
            f"kl_beta: {self.kl_beta}, grpo_group_size: {self.grpo_group_size}, "
            f"temperature: {self.temperature}, top_k: {self.top_k}, top_p: {self.top_p}"
        )
        self.myprint(f"self.model: {self.model.device}")

    def _set_signature_columns_if_needed(self):
        self._signature_columns = ['pixel_values', 'input_ids', 'attention_mask', 'position_ids', 'image_flags', 'past_key_values', 'labels', 'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict', 'statistics', 'loss_weight', 'loss_reduction_all_gather', 'target_masks', 'labels', 'label_ids', 'label']
        self._signature_columns = self._signature_columns + ["conversations"]
        
    def myprint(self, text):
        if self.is_world_process_zero():
            print(f"Step {self.state.global_step}: {text}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        tt_start_token_id =  self.tokenizer.encode(SEG_START_TOKEN)[-1]
        tt_end_token_id =  self.tokenizer.encode(SEG_END_TOKEN)[-1]
        tt_id_first =  self.tokenizer.encode(SEG_TOKEN_TEMPLATE.format(0))[-1]

        def _check_tt_ids_valid(tt_ids):
            for tt_id in tt_ids:
                if tt_id < tt_id_first or tt_id > tt_id_first+COODBOOK_SIZE-1:
                    return 0
            return 1

        def reward_tt_format(completion_id, tt_start_token_id, tt_end_token_id, NUM_HIMT_TOKENS):
            tt_start_idxs = (completion_id == tt_start_token_id).nonzero(as_tuple=True)[0]
            tt_end_idxs = (completion_id == tt_end_token_id).nonzero(as_tuple=True)[0]

            r_tt_start = 1 if len(tt_start_idxs) == 1 else 0
            r_tt_end = 1 if len(tt_end_idxs) == 1 else 0
            
            tt_start_idx, tt_end_idx = None, None
            if (len(tt_start_idxs) != 1) or (len(tt_end_idxs) != 1):
                r_tt_count, r_tt_id_valid = 0, 0
                token_num_between = 0
            else:
                tt_start_idx = tt_start_idxs[0]
                tt_end_idx = tt_end_idxs[0]
                token_num_between = (tt_end_idx - tt_start_idx - 1).item()
                if token_num_between > 0 and token_num_between <= NUM_HIMT_TOKENS:
                    r_tt_count = 1
                    r_tt_id_valid = _check_tt_ids_valid(completion_id[tt_start_idx+1:tt_end_idx])
                else:
                    r_tt_count, r_tt_id_valid = 0, 0
                
            rewards = (r_tt_start, r_tt_end, r_tt_count, r_tt_id_valid)
            return rewards, tt_start_idx, tt_end_idx, token_num_between

        current_step = self.state.global_step
        self.myprint(f"\n ---- compute_loss start {current_step} ------ \n")
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        device = self.accelerator.device

        # Prepare chat config and prompts
        chat_config = dict(
            max_new_tokens=256,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            return_ids=True,
        )
        assert chat_config['return_ids'] == True

        pixel_values = inputs['pixel_values']
        conversations = inputs['conversations']
        prompts = []
        mask_only_with_adaptive_len_template = [
            "Segment {} by adaptive length.",
            "Create a mask for {} by adaptive length.",
            "Generate a mask for {} by adaptive length.",
            "Do segmentation for {} by adaptive length.",
            "Please give the mask for {} by adaptive length.",
            "What is the mask for {} by adaptive length?",
            "Can you segment {} by adaptive length?"
        ]
        for i, conv in enumerate(conversations):
            if "<ref>" in conv:
                obj = "<ref>" + conv.split("<ref>")[1].split("</ref>")[0] + "</ref>"
            elif "<box>" in conv:
                obj = "<box>" + conv.split("<box>")[1].split("</box>")[0] + "</box>"
            else:
                raise NotImplementedError(f"Unsupported conversation format: {conv}")
            prompt = mask_only_with_adaptive_len_template[i % len(mask_only_with_adaptive_len_template)].format(obj)
            prompts.append(prompt)

        prompts = [prompt for prompt in prompts for _ in range(self.grpo_group_size)]

        pixel_values = pixel_values.repeat_interleave(self.grpo_group_size, dim=0)

        model.eval()
        ret = model.batch_chat(self.tokenizer, pixel_values,
                              num_patches_list=[1] * len(prompts),
                              questions=prompts,
                              generation_config=chat_config)
        model.train()
        responses_ret, query_ids_ret, completion_ids_ret = ret
        self.myprint(f"len(responses_ret), query_ids_ret.shape, completion_ids_ret.shape: {len(responses_ret)}, {query_ids_ret.shape}, {completion_ids_ret.shape}")

        prompt_completion_ids = torch.cat([query_ids_ret, completion_ids_ret], dim=1)
        completion_max_len = completion_ids_ret.size(1)
        is_eos = completion_ids_ret == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        # completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()  # Unused

        image_flags = torch.tensor([1] * len(query_ids_ret), device=device)

        other_inputs = {
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "return_dict": True,
        }

        def get_per_token_logps_part1(model, input_ids, **kwargs):
            ret = model(input_ids=input_ids, **kwargs)
            logits = ret.logits
            input_ids_ = input_ids[:, 1:]
            logits_ = logits[:, :-1, :]
            return input_ids_, logits_

        all_input_ids, all_logits = get_per_token_logps_part1(model, prompt_completion_ids, **other_inputs)
        completion_logits = all_logits.contiguous()[:, -completion_max_len:, :]
        completion_ids = all_input_ids.contiguous()[:, -completion_max_len:]
        assert ids_are_same(completion_ids, completion_ids_ret), "completion_ids and completion_ids_ret are not the same!"

        tt_format_rewards_list = []
        tt_length_rewards = []
        tt_probs = torch.zeros(completion_logits.size(0), NUM_HIMT_TOKENS, COODBOOK_SIZE, device=device, dtype=completion_logits.dtype)
        valid_mask = torch.zeros(completion_logits.size(0), dtype=torch.bool, device=device)
        for i, (completion_id, completion_logit) in enumerate(zip(completion_ids, completion_logits)):
            rewards, tt_start_idx, tt_end_idx, token_num_between = reward_tt_format(completion_id, tt_start_token_id, tt_end_token_id, NUM_HIMT_TOKENS)
            invalid_flag = min(rewards) < 0.5
            tt_length_rewards.append(token_num_between)

            if invalid_flag:
                self.myprint(f"invalid tt tokens in sample {i}, use dummy tt ids and tt logits!")
                dummy_ids = torch.arange(tt_id_first, tt_id_first + NUM_HIMT_TOKENS, device=device)
                dummy_logits = completion_logit[0:NUM_HIMT_TOKENS, :]
                # tt_ids and tt_logits are not used, so not appended
            else:
                tt_prob_end = (torch.softmax(completion_logit[tt_start_idx+2:tt_end_idx+1, :], dim=-1)[:, tt_end_token_id] * 256).long()
                tt_prob_end = torch.cat([tt_prob_end, torch.zeros(NUM_HIMT_TOKENS-tt_prob_end.size(0), dtype=torch.long, device=device)], dim=0)
                onehot = torch.zeros(token_num_between, COODBOOK_SIZE, dtype=completion_logit.dtype, device=device)
                onehot = onehot.scatter_(-1, completion_id[tt_start_idx+1:tt_end_idx].unsqueeze(-1)-tt_id_first, 1.0)
                tt_probs[i, :token_num_between, :] = onehot
                valid_mask[i] = True
            tt_format_rewards_list.append(rewards)
        self.myprint(f"tt_format_rewards_list: {tt_format_rewards_list}")

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            image_src = model.convert_image_to_sam_input(pixel_values[:1, ...]) * 255
            image_src = unwrapped_model.mask_decoder.sam.preprocess(image_src)
            image_embedding = unwrapped_model.mask_decoder.sam.image_encoder(image_src).repeat_interleave(self.grpo_group_size, dim=0)
            mask_images = unwrapped_model.mask_decoder.decode_prob(tt_probs[:, :NUM_HIMT_TOKENS, :], image_embedding=image_embedding).mean(dim=1, keepdim=False)

        target_masks = inputs['target_masks']
        target_masks = torch.repeat_interleave(target_masks, self.grpo_group_size, dim=0)

        ious = reward_iou(mask_images, target_masks, valid_mask)

        for i, rewards in enumerate(tt_format_rewards_list):
            if min(rewards) < 0.5:
                ious[i] = 0.0

        rewards = []
        for rewards_tt_format, iou, tt_length_reward in zip(tt_format_rewards_list, ious, tt_length_rewards):
            reward = sum(rewards_tt_format) * 0.1 + (NUM_HIMT_TOKENS-tt_length_reward) * self.length_weight + iou
            rewards.append(reward)
        rewards = torch.tensor(rewards, device=device)

        self.myprint(f"ious: {ious}")
        self.myprint(f"rewards: {rewards}")

        mean_grouped_rewards = rewards.view(-1, self.grpo_group_size).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.grpo_group_size).std(dim=1)
        self.myprint(f"before normalize: mean_grouped_rewards: {mean_grouped_rewards}, std_grouped_rewards: {std_grouped_rewards}")
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.grpo_group_size, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.grpo_group_size, dim=0)

        if self.ref_model is not None:
            with torch.inference_mode():
                ref_all_input_ids, ref_all_logits = get_per_token_logps_part1(self.ref_model, prompt_completion_ids, **other_inputs)
                ref_completion_logits = ref_all_logits.contiguous()[:, -completion_max_len:, :]
                ref_completion_ids = ref_all_input_ids.contiguous()[:, -completion_max_len:]
            assert ids_are_same(ref_completion_ids, completion_ids) and ids_are_same(ref_all_input_ids, all_input_ids), "ids are not the same!"

            def get_per_token_logps_part2(input_ids, logits):
                per_token_logps = []
                for logits_row, input_ids_row in zip(logits, input_ids):
                    log_probs = logits_row.log_softmax(dim=-1)
                    token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                    per_token_logps.append(token_log_prob)
                per_token_logps = torch.stack(per_token_logps, dim=0)
                return per_token_logps

            ref_per_token_logps_for_completion = get_per_token_logps_part2(ref_completion_ids, ref_completion_logits)
            per_token_logps_for_completion = get_per_token_logps_part2(completion_ids, completion_logits)

            per_token_kl = torch.exp(ref_per_token_logps_for_completion - per_token_logps_for_completion) - (ref_per_token_logps_for_completion - per_token_logps_for_completion) - 1

            self.myprint(f"per_token_kl statistics: shape: {per_token_kl.shape}, mean: {per_token_kl.mean()}, min: {per_token_kl.min()}, max: {per_token_kl.max()}")

        # Only keep the normalized advantage branch
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4 + 0.1)
        self.myprint(f"advantages: {advantages}")

        per_token_loss = torch.exp(per_token_logps_for_completion - per_token_logps_for_completion.detach()) * advantages.unsqueeze(1)
        if self.ref_model is not None:
            per_token_loss = -(per_token_loss - self.kl_beta * per_token_kl)
        else:
            per_token_loss = - per_token_loss
        loss = per_token_loss.mean()
        loss = torch.clamp(loss, max=1.0)
        self.myprint(f"step {current_step} loss: {loss}")

        return loss
