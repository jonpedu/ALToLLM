from typing import Mapping, Text, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

class HiMTLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        loss_config = config.losses
        self.config = config
        
        # Main loss weights
        self.bce_weight = loss_config.get("bce_weight", 2.0)
        self.dice_weight = loss_config.get("dice_weight", 0.5)
        self.iou_weight = loss_config.get("iou_weight", 1.0)
        self.quantizer_weight = loss_config.get("quantizer_weight", 1.0)
        
        # Smoothing factor for dice loss
        self.smooth = 1e-5


        ####################################
        self.reconstruction_weight = loss_config.reconstruction_weight
        # Add edge loss weight
        self.edge_weight = loss_config.get("edge_weight", self.reconstruction_weight*0.2)
        self.length_weight = loss_config.get("length_weight", 0.1)
        
        # Create Sobel kernels for edge detection with float32 dtype
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))


    def compute_edge_maps(self, inputs):
        """Compute edge maps using multi-scale Sobel operators"""
        device = inputs.device
        
        # Convert to grayscale if input is RGB
        if inputs.shape[1] == 3:
            inputs_gray = 0.2989 * inputs[:, 0:1] + 0.5870 * inputs[:, 1:2] + 0.1140 * inputs[:, 2:3]
        else:
            inputs_gray = inputs

        # Multi-scale edge detection
        kernel_sizes = [3, 5]  # Multiple kernel sizes for wider edges
        weights = [1.0, 0.5]  # Weights for different scales
        
        total_edge_map = 0
        for size, weight in zip(kernel_sizes, weights):
            # Create larger Sobel kernels
            center = size // 2
            sigma = size / 3.0
            x = torch.arange(size, dtype=torch.float32, device=device) - center
            gaussian = torch.exp(-(x ** 2) / (2 * sigma ** 2))
            gaussian = gaussian / gaussian.sum()
            
            # Create Sobel kernels
            sobel = torch.arange(-(size//2), size//2 + 1, dtype=torch.float32, device=device)
            sobel = sobel / sobel.abs().max()
            
            # Construct 2D kernels
            kernel_x = gaussian.view(1, -1) * sobel.view(-1, 1)
            kernel_y = kernel_x.t()
            
            # Reshape kernels for conv2d
            kernel_x = kernel_x.view(1, 1, size, size)
            kernel_y = kernel_y.view(1, 1, size, size)

            # Compute gradients
            grad_x = F.conv2d(inputs_gray, kernel_x, padding=size//2)
            grad_y = F.conv2d(inputs_gray, kernel_y, padding=size//2)
            
            # Compute edge map for current scale
            edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
            
            # Add to total edge map with weight
            total_edge_map += weight * edge_map

        # Apply soft thresholding to make edges more prominent
        threshold = 0.1
        total_edge_map = torch.sigmoid((total_edge_map - threshold) * 5)
        
        # Normalize edge map to [0, 1]
        total_edge_map = (total_edge_map - total_edge_map.min()) / (total_edge_map.max() - total_edge_map.min() + 1e-8)
        
        # Apply dilation to make edges wider
        # kernel_size = 5
        # dilate_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
        # total_edge_map = F.conv2d(total_edge_map, dilate_kernel, padding=kernel_size//2)
        # total_edge_map = (total_edge_map - total_edge_map.min()) / (total_edge_map.max() - total_edge_map.min() + 1e-8)
        
        return total_edge_map
    
    def compute_edge_loss(self, targets: torch.Tensor, reconstructions: torch.Tensor, edge_weights: torch.Tensor=None) -> torch.Tensor:
        """Compute edge loss using Sobel operators."""
        # Convert to grayscale if input is RGB
        if targets.shape[1] == 3:
            inputs_gray = 0.2989 * targets[:, 0:1] + 0.5870 * targets[:, 1:2] + 0.1140 * targets[:, 2:3]
            recon_gray = 0.2989 * reconstructions[:, 0:1] + 0.5870 * reconstructions[:, 1:2] + 0.1140 * reconstructions[:, 2:3]
        else:
            inputs_gray = targets
            recon_gray = reconstructions

        edge_weights = edge_weights if edge_weights is not None else 1.0
        # Compute multi-scale edge maps
        edge_loss = 0
        # kernel_sizes = [3, 5, 7]  # 多尺度的 Sobel 算子
        kernel_sizes = [3,]  # 多尺度的 Sobel 算子
        weights = [1.0, 0.5, 0.25]  # 不同尺度的权重

        for size, weight in zip(kernel_sizes, weights):
            # Create Sobel kernels for current size
            kernel_x = torch.ones((size, size), dtype=torch.float32, device=targets.device)
            kernel_x[:, size//2:] = -1
            kernel_x = kernel_x.view(1, 1, size, size)
            
            kernel_y = kernel_x.transpose(2, 3)
            
            # Compute gradients
            input_grad_x = F.conv2d(inputs_gray, kernel_x, padding=size//2)
            input_grad_y = F.conv2d(inputs_gray, kernel_y, padding=size//2)
            recon_grad_x = F.conv2d(recon_gray, kernel_x, padding=size//2)
            recon_grad_y = F.conv2d(recon_gray, kernel_y, padding=size//2)

            # Compute edge maps
            input_edges = torch.sqrt(input_grad_x ** 2 + input_grad_y ** 2 + 1e-6)
            recon_edges = torch.sqrt(recon_grad_x ** 2 + recon_grad_y ** 2 + 1e-6)

            # Add weighted loss for current scale
            edge_loss += weight * (
                0.5 * F.mse_loss(input_edges, recon_edges, reduction="none")*edge_weights + 
                0.5 * F.smooth_l1_loss(input_edges, recon_edges, beta=0.4, reduction="none")*edge_weights
            ).mean()

        return edge_loss

    def dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss using probabilities
        Args:
            probs: prediction tensor after sigmoid [B, C, H, W]
            targets: target tensor [B, C, H, W]
        """
        # Flatten the tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice

    def bce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary Cross Entropy Loss using logits
        Args:
            logits: prediction tensor before sigmoid [B, C, H, W]
            targets: target tensor [B, C, H, W]
        """
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
    
    def _forward_generator_mse(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int,
                           edge_weight: float = 0.0
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        #resize input to the same size as reconstruction
        # inputs = F.interpolate(inputs, size=reconstructions.shape[-2:], mode='bilinear')
        reconstructions = F.interpolate(reconstructions, size=inputs.shape[-2:], mode='bilinear')
        
        if edge_weight > 0.0:   
            with torch.no_grad():
                edge = self.compute_edge_maps(inputs) 
        else:
            edge = 0

        # Calculate BCE loss
        bce_loss = self.bce_loss(reconstructions, inputs)
        
        # Calculate Dice loss
        probs = torch.sigmoid(reconstructions)
        dice_loss = self.dice_loss(probs, inputs)

        mse_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        edge_loss = self.compute_edge_loss(inputs, reconstructions, edge_weight)
        mse_edge_enhanced_loss = F.mse_loss(inputs * (1 + edge* edge_weight), reconstructions * (1 + edge* edge_weight), reduction="mean")

        quantizer_loss = extra_result_dict["quantizer_loss"]
        total_loss = mse_edge_enhanced_loss * self.reconstruction_weight + \
            self.edge_weight * edge_loss + \
            self.quantizer_weight * quantizer_loss + \
            self.length_weight * extra_result_dict["length_loss"] + \
            self.bce_weight * bce_loss + \
            self.dice_weight * dice_loss
        
        loss_dict = dict(
            mse_loss = mse_loss,
            edge_loss = edge_loss,
            total_loss = total_loss,
            reconstruction_loss = mse_edge_enhanced_loss,
            quantizer_loss = (self.quantizer_weight * quantizer_loss),
            commitment_loss = extra_result_dict["commitment_loss"],
            codebook_loss = extra_result_dict["codebook_loss"],
            length_loss = extra_result_dict["length_loss"],
            bce_loss = bce_loss,
            dice_loss = dice_loss
        )

        return total_loss, loss_dict
    
    @autocast('cuda', enabled=False)
    def forward(self,
                targets: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [0, 1].
        # if targets.shape[1] == 3:
        #     targets = targets.mean(1, keepdim=True)
        if reconstructions.shape[1] == 1:
            reconstructions = reconstructions.repeat(1, 3, 1, 1)
            
        targets = targets.float()
        reconstructions = [x.float() for x in reconstructions] if isinstance(reconstructions, list) else reconstructions.float()

        lengths_to_keep = extra_result_dict["lengths_to_keep"].to(targets.device)  #(B,)
        # print(lengths_to_keep)
        B = lengths_to_keep.shape[0]
        total_loss = 0
        total_loss_dict = {}
        for i, l in enumerate(lengths_to_keep):
            l = float(l.item())
            # W = max(16, round(0.25*(l**2)))
            W = max(16, round((l/32)**1.5*256))
            H = W
            _reco = reconstructions[i:i+1,...]
            _target = targets[i:i+1,...]
            # extra_result_dict["lengths_to_keep"] = lengths_to_keep[i:i+1,...]

            if 0 and W == 256:
                loss, loss_dict = self._forward_generator(_target, _reco, extra_result_dict, global_step)
            else:
                #reszie input to the same size as reconstruction
                # _target = F.interpolate(_target, size=(H,W), mode='bilinear')
                edge_weight =4# (min(1,l/64))*4# if self.config.model.use_vae else 4
                loss, loss_dict = self._forward_generator_mse(_target, _reco, extra_result_dict, global_step, edge_weight=edge_weight)

            total_loss += loss/B
            for k, v in loss_dict.items():
                total_loss_dict[k] = total_loss_dict.get(k, 0) + v/B
        # total_loss, total_loss_dict = self._forward_generator_mse(targets, reconstructions, extra_result_dict, global_step)
        return total_loss, total_loss_dict
    

    @autocast('cuda', enabled=False)
    def forward_only_mse(self,
                targets: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator"):
        targets = targets.float()  # [B, 3, 256, 256]
        if targets.shape[1] == 3:
            targets = targets.mean(1, keepdim=True)

        reconstructions = reconstructions.float()  # [B, 3, 256, 256]
        if reconstructions.shape[1] == 3:
            reconstructions = reconstructions.mean(1, keepdim=True)

        #compute mse loss between targets and reconstructions
        mse_loss = 10*F.mse_loss(reconstructions, targets, reduction='mean')

        # Combine losses
        quantizer_loss = extra_result_dict["quantizer_loss"]
        total_loss = (1.0 * mse_loss + 
                      self.quantizer_weight * quantizer_loss)
        
        # Store losses
        total_loss_dict = {
            'mse_loss': mse_loss,
            'bce_loss': 0,
            'dice_loss': 0,
            'iou_loss': 0,
            'mse_loss': mse_loss,
            'quantizer_loss': (self.quantizer_weight * quantizer_loss),
            'total_loss': total_loss
        }
        
        return total_loss, total_loss_dict

