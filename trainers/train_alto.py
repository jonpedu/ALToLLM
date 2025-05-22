import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from trainers.ema_pytorch import EMA
from trainers.trainer import DistributedTrainerBase
from trainers.helper import cprint, init_seeds
from net.alto import ALTo
# from dataset.mydataset import SupervisedDataset as Dataset
from trainers.dataset import SupervisedDataset as Dataset
from trainers.loss_alto import HiMTLoss
import math
from pathlib import Path
from trainers.viz_utils import make_viz_from_samples
import matplotlib.pyplot as plt
import json
import numpy as np

# filter warnings from all modules
import warnings
warnings.filterwarnings("ignore")

class TrainProcess(DistributedTrainerBase):
    def __init__(self, rank, local_rank, opt):
        super().__init__()
        if rank == 0:
            cprint('#### [TrainTempolate] Start main Process. pid=%d' % os.getpid(), 'red')
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = opt.world_size
        self.start_epoch = 0
        self.start_step = 0
        self.opt = opt
        self.current_lr = opt.learning_rate
        self.dtype = self.set_dtype(opt.dtype)
        self.start_time = time.time()

    def run(self):
        """
        Main entry point for the distributed training process.
        Initializes the training environment and starts the training loop.
        """
        cprint(f'#### Start run rank {self.rank} (local_rank={self.local_rank}) Process. pid={os.getpid()}', 'red')
        torch.cuda.set_device(self.local_rank)
        # Initialize seeds for reproducibility
        init_seeds(self.opt.seed + self.rank)
        # Initialize the distributed environment
        self.init_distributed_env()
        # Initialize necessary components like dataloaders, model, and tensorboard
        # self.set_meter(['grad_norm', 'iou_loss', 'bce_loss', 'dice_loss', 'quantizer_loss','mse_loss','total_loss'])
        self.set_meter(['grad_norm', 'quantizer_loss','mse_loss', 'edge_loss', 'reconstruction_loss', 'total_loss',"length_loss","bce_loss","dice_loss"])
        self.set_dataprovider(self.opt)
        self.set_model_and_loss() 
        if self.rank == 0:
            logpath = os.path.join(self.opt.model_save_dir, 'logs')
            self.sw = SummaryWriter(logpath)
            # self.add_model_graph(self.model, (512,1))

        self.verify_weights_sync()
        # Optionally test the model before training begins
        if self.opt.test_before_train and (self.opt.test_distributed or self.rank == 0):
            if self.opt.vis_all and not self.opt.model.vq_model.finetune_decoder:
                # for length in [2,4,8,16,32]:
                #     for use_vae in [True, False]:
                for length in [32]:
                    for use_vae in [True]:
                        self.model.config.model.test_length = length
                        self.model.config.model.use_vae = use_vae
                        print(f"test length: {length}, use_image: {use_vae}")
                        self.test(self.start_step)
            else:
                self.test(self.start_step)
        # Begin the training loop
        self.train()

    def init_distributed_env(self):
        """
        Initializes the distributed training environment with the NCCL backend.
        """
        cprint(f'[rank-{self.rank}] Initializing distributed environment...', 'cyan')
        torch.distributed.init_process_group(backend='nccl', world_size=self.opt.world_size, rank=self.rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        if hasattr(self.opt, 'cuda_deterministic') and self.opt.cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def set_dataprovider(self, opt):
        cprint(f'[rank-{self.rank}] Setting up dataloader...', 'cyan')

        distributed = self.opt.world_size > 1
        self.trainset = Dataset(config=opt, is_train=True)
        print("self.trainset:",self.trainset) 
        train_sampler = DistributedSampler(self.trainset, rank=self.rank, num_replicas=self.world_size,
                                           shuffle=True) if distributed else None
        self.train_loader = DataLoader(self.trainset,
                                       collate_fn=None,
                                       num_workers=opt.dataset.params.num_workers_per_gpu,
                                       shuffle=False if distributed else True,
                                       sampler=train_sampler,
                                       batch_size=opt.batch_size_per_gpu,
                                       pin_memory=False,
                                       persistent_workers=True,
                                       drop_last=True,
                                       prefetch_factor=2)

        if opt.test_distributed or self.rank == 0:
            self.testset = Dataset(config=opt, is_train=False)
            print("self.testset:",self.testset) 
            test_sampler = DistributedSampler(self.testset, rank=self.rank, num_replicas=self.world_size,
                                              shuffle=False) if distributed else None
            self.test_loader = DataLoader(self.testset,
                                          collate_fn=None,
                                          num_workers=opt.dataset.params.num_workers_per_gpu,
                                          shuffle=False,
                                          sampler=test_sampler,
                                          batch_size=opt.batch_size_per_gpu,
                                          pin_memory=False,
                                          persistent_workers=True,
                                          drop_last=True)

    def set_model_and_loss(self):
        cprint(f'[rank-{self.rank}] Setting up model...', 'cyan')
        self.model = ALTo(self.opt).to(self.device)
        self.loss_fn = HiMTLoss(self.opt).to(self.device)
        
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.opt.learning_rate,
            weight_decay=self.opt.weight_decay,
            betas=[self.opt.beta1, self.opt.beta2]
        )
        
        self.lr_scheduler = self.create_scheduler(
            self.optim,
            schedule_type=self.opt.lr_schedule,
            warm_steps=self.opt.warm_steps,
            total_steps=self.opt.total_steps,
            min_lr=self.opt.min_lr,
            max_lr=self.opt.learning_rate
        )

        if self.rank == 0 and self.opt.use_ema:
            self.model_ema = EMA(
                self.model,
                beta=self.opt.ema_beta,  # exponential moving average factor
                update_after_step=0,  # only after this number of .update() calls will it start updating. default=100
                update_every=self.opt.ema_update_every,
                # how often to actually update, to save on compute (updates every 10th .update() call)
                start_step=1000000,
                ma_device=self.device,
            )

        self.load_ckpt_if_exist(self.opt.experiment.init_weight, verbose=True)
        self.configure_model_gradients()

        if self.opt.world_size > 1:
            # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                             find_unused_parameters=self.opt.find_unused_parameters)
                    
    def configure_model_gradients(self):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        model.requires_grad_(False)
        if self.opt.model.vq_model.finetune_encoder:
            # model.encoder.length_encoder.apply(model._init_weights)
            model.encoder.requires_grad_(True)
            model.quantize.requires_grad_(True)
            model.encoder.length_encoder.requires_grad_(False)
        if self.opt.model.vq_model.finetune_length:
            model.encoder.length_encoder.apply(model._init_weights)
            model.encoder.length_encoder.requires_grad_(True)
        if self.opt.model.vq_model.finetune_decoder:
            model.decoder.requires_grad_(True)
        if hasattr(model, "vae"):
            model.vae.requires_grad_(False)
        if hasattr(model, "sam"):
            model.sam.requires_grad_(False)
            # model.sam.prompt_encoder.requires_grad_(False)
            # model.sam.mask_decoder.requires_grad_(False)
            model.sam.image_encoder.requires_grad_(False)
        
        if self.rank == 0:
            self.count_parameter(model, tag='')
            if hasattr(model, "vae"):
                self.count_parameter(model.vae, tag='  - vae')
            self.count_parameter(model.encoder, tag='  - encoder')
            self.count_parameter(model.quantize, tag='  - quantize')
            self.count_parameter(model.decoder, tag='  - decoder')
            if hasattr(model,'sam'):
                self.count_parameter(model.sam, tag='  - sam')
                self.count_parameter(model.sam.image_encoder, tag='  - image_encoder')
                # self.count_parameter(model.sam.prompt_encoder, tag='  - prompt_encoder')
                # self.count_parameter(model.sam.mask_decoder, tag='  - mask_decoder')

    def preprocess(self, mb):
        for name, x in mb.items():
            if name in ['mask', 'image_src']:
                x = x.to(self.device)
            mb[name] = x
        return mb

    def do_optimize(self, mb, step, scaler):
        self.model.train()
        opt = self.opt
        gradient_accu_steps = opt.gradient_accumulation_steps
        if opt.world_size > 1:
            self.model.require_backward_grad_sync = False if step % gradient_accu_steps != 0 else True

        with autocast('cuda', enabled=self.opt.enable_amp, dtype=self.dtype):
            masks = mb['mask']
            image_src = mb['image_src']
            reconstructed_images, extra_results_dict = self.model(masks, image_src = image_src)
            total_loss, loss_dict = self.loss_fn(masks, reconstructed_images, extra_results_dict,step,mode="generator")


        scaler.scale(total_loss / gradient_accu_steps).backward()
        if step % gradient_accu_steps == 0:
            scaler.unscale_(self.optim)
            loss_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                     self.opt.max_grad_norm,  # 例如 1.0
                                                     error_if_nonfinite=False)  # 防止出现inf/nan时报错                
            scaler.step(self.optim)
            scaler.update()
            self.optim.zero_grad()
            self.current_lr = self.lr_scheduler.step()
            
        if opt.world_size > 1:
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    torch.distributed.all_reduce(v.detach().to(self.device), op=torch.distributed.ReduceOp.AVG)
        self.update_meter(loss_dict)

        return loss_dict

    def train(self):
        opt = self.opt
        rank = self.rank
        world_size = self.world_size
        grad_accu_steps = opt.gradient_accumulation_steps

        grad_scaler = GradScaler('cuda', enabled=opt.enable_amp)
        self.model.train()
        torch.cuda.empty_cache()

        mb_size = self.opt.batch_size_per_gpu
        t0 = time.time()
        step = self.start_step if hasattr(self, 'start_step') else 0
        grad_step = step * grad_accu_steps  # when grad_accu_steps==1, grad_step is step
        # start_epoch = step // (len(self.trainset) // opt.batch_size) + 1
        start_epoch = 0
        for epoch in range(start_epoch, opt.max_epochs):
            for mb in self.train_loader:
                # print('rank=%d step=%d' % (rank, step))
                # torch.distributed.barrier(async_op=False)  # sync for training
                # forward
                grad_step += 1
                step = grad_step // grad_accu_steps
                mb = self.preprocess(mb)
                mb_losses = self.do_optimize(mb, grad_step, grad_scaler)
                if hasattr(self, 'model_ema'):  # only rank-0 own model_ema
                    self.model_ema.update()

                if rank == 0 and grad_step % (opt.print_freq * grad_accu_steps) == 0:
                    # torch.cuda.synchronize()  # sync for logging
                    now = time.time()
                    ct = now - t0
                    speed = opt.print_freq * grad_accu_steps * world_size * mb_size / ct
                    speed_iter = float(opt.print_freq) / ct
                    str = 'epoch %d ([Step%d]x[Bs%dx%d]x[GPU%d]) took %0.1fs(%0.2fh) %0.1fimgs/s %0.2fiter/s lr=%0.6f' % \
                          (epoch, step, mb_size, grad_accu_steps, world_size, ct,
                           (now - self.start_time) / 3600, speed, speed_iter, self.current_lr)
                    str += self.get_meter(str=True)
                    cprint(str)

                    self.sw.add_scalar("train/lr", self.current_lr, step)
                    for k, v in self.get_meter().items():
                        self.sw.add_scalar("train/%s" % k, v, step)

                    self.reset_meter()
                    self.log_memory_stats(step)
                    t0 = time.time()

                if (opt.test_distributed or rank == 0) and grad_step % (opt.test_freq * grad_accu_steps) == 0:
                    test_loss = self.test(step)
                    self.reset_meter()  # maybe used in test
                    self.model.train()
                    if rank == 0:
                        self.save_model(epoch, step, test_loss, opt.model_save_dir, max_time_not_save=0 * 60)

                # Check for NaN loss
                if rank == 0 and torch.isnan(mb_losses['total_loss']).any():
                    cprint('NaN loss detected! Training terminated.', 'red', attrs=['blink'])
                    # Save model state for debugging
                    self.save_model(epoch, step, float('nan'), opt.model_save_dir, 
                                  max_time_not_save=0, 
                                  suffix='_nan_checkpoint')
                    # Ensure all processes terminate
                    if world_size > 1:
                        torch.distributed.destroy_process_group()
                    raise RuntimeError("NaN loss detected. Training terminated.")

    @torch.no_grad()
    def test(self, step):
        self.model.eval()

        t0 = time.time()
        cnt, loss_sum = 0, 0
        max_save_images = 32 if self.opt.model.vq_model.finetune_decoder else 1024
        mean_iou_list = []
        mean_biou_list = []
        mean_mse_list = []
        lengths_to_keep_list = []
        mask_count_list = []  # Add list to collect mask_count data
        for mb in self.test_loader:
            mb = self.preprocess(mb)
            masks = mb['mask']
            image_src = mb['image_src']
            mask_count = mb['mask_count']
            B = masks.shape[0]
            with autocast('cuda', enabled=self.opt.enable_amp, dtype=self.dtype):
                reconstructed_images, extra_results_dict = self.model(masks, image_src = image_src)
                # Calculate losses using the loss class
                total_loss, loss_dict = self.loss_fn(masks, reconstructed_images, extra_results_dict,step,mode="generator")
            if cnt < max_save_images and self.rank == 0:
                mean_iou, mean_biou, mean_mse = self.reconstruct_images(
                    masks,
                    reconstructed_images,
                    extra_results_dict,
                    image_src,
                    step,
                    self.opt.model_save_dir,
                    cnt
                )
                mean_iou_list.append(mean_iou)
                mean_biou_list.append(mean_biou)
                mean_mse_list.append(mean_mse)
                lengths_to_keep_list.append(extra_results_dict['lengths_to_keep'])
                mask_count_list.append(mask_count)  # Collect mask_count data
            loss_sum += total_loss * B
            cnt += B

            if cnt >= max_save_images:
                break
        if len(mean_iou_list) > 0:
            mean_iou = sum(mean_iou_list) / len(mean_iou_list)
            mean_biou = sum(mean_biou_list) / len(mean_biou_list)
            mean_mse = sum(mean_mse_list) / len(mean_mse_list)
            print(f"mean_iou: {mean_iou:.4f}, mean_biou: {mean_biou:.4f}, mean_mse: {mean_mse:.4f}")
            
            # Calculate length distribution statistics
            all_lengths = torch.cat(lengths_to_keep_list)
            all_mask_counts = torch.cat(mask_count_list)  # Concatenate all mask counts
            
            # Calculate correlation between mask_count and lengths_to_keep
            correlation = torch.corrcoef(torch.stack([all_mask_counts.float().cpu(), all_lengths.float().cpu()]))[0, 1].item()
            
            # Save correlation data to file
            correlation_data = {
                'step': step,
                'correlation': correlation,
                'mask_counts': all_mask_counts.cpu().numpy().tolist(),
                'lengths': all_lengths.cpu().numpy().tolist()
            }
            correlation_path = os.path.join(self.opt.model_save_dir, "train_images", f"correlation_data_step_{step:04}.json")
            with open(correlation_path, 'w') as f:
                json.dump(correlation_data, f, indent=4)
            
            # Plot scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(all_mask_counts.cpu().numpy(), all_lengths.cpu().numpy(), alpha=0.5)
            plt.title(f'Mask Count vs Length Distribution (Correlation: {correlation:.4f})')
            plt.xlabel('Mask Count')
            plt.ylabel('Length')
            plt.grid(True, alpha=0.3)
            
            # Add correlation coefficient as text
            plt.text(0.95, 0.95, f'Correlation: {correlation:.4f}',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save plot
            plot_path = os.path.join(self.opt.model_save_dir, "train_images", f"correlation_plot_step_{step:04}.png")
            plt.savefig(plot_path)
            plt.close()
            
            print(f"Correlation between mask_count and lengths_to_keep: {correlation:.4f}")
            
            mean_length = all_lengths.float().mean().item()
            std_length = all_lengths.float().std().item()
            
            # Calculate length distribution entropy
            length_counts = torch.bincount(all_lengths, minlength=33)  # Assuming max length is 32
            length_probs = length_counts.float() / length_counts.sum()
            length_entropy = -torch.sum(length_probs * torch.log2(length_probs + 1e-10)).item()
            
            # Save length distribution data to file
            length_dist_data = {
                'step': step,
                'length_probs': length_probs.cpu().numpy().tolist(),
                'mean_length': mean_length,
                'std_length': std_length,
                'length_entropy': length_entropy
            }
            length_dist_path = os.path.join(self.opt.model_save_dir, "train_images", f"length_dist_data_step_{step:04}.json")
            with open(length_dist_path, 'w') as f:
                json.dump(length_dist_data, f, indent=4)
            
            # Calculate entropy statistics
            entropy_list = []
            for lengths in lengths_to_keep_list:
                counts = torch.bincount(lengths, minlength=33)
                probs = counts.float() / counts.sum()
                entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
                entropy_list.append(entropy)
            
            mean_entropy = sum(entropy_list) / len(entropy_list)
            std_entropy = torch.tensor(entropy_list).std().item()
            
            print(f"Length Statistics:")
            print(f"  Mean Length: {mean_length:.2f}")
            print(f"  Length Std: {std_length:.2f}")
            print(f"  Overall Length Entropy: {length_entropy:.2f}")
            print(f"  Mean Entropy: {mean_entropy:.2f}")
            print(f"  Entropy Std: {std_entropy:.2f}")
            
            # Save length distribution plot with proportions
            plt.figure(figsize=(10, 6))
            plt.hist(all_lengths.cpu().numpy(), bins=33, range=(0, 33), alpha=0.7, weights=np.ones_like(all_lengths.cpu().numpy()) / len(all_lengths))
            plt.title(f'Length Distribution')
            plt.xlabel('Length')
            plt.ylabel('Proportion')
            plt.grid(True, alpha=0.3)
            
            # Add statistics as text
            stats_text = f'Mean: {mean_length:.2f}\nStd: {std_length:.2f}\nEntropy: {length_entropy:.2f}'
            plt.text(0.95, 0.95, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save plot
            plot_path = os.path.join(self.opt.model_save_dir, "train_images", f"length_dist_step_{step:04}.png")
            plt.savefig(plot_path)
            plt.close()
            
            # 将指标添加到TensorBoard
            if self.rank == 0:
                self.sw.add_scalar("test/mean_iou", mean_iou, step)
                self.sw.add_scalar("test/mean_biou", mean_biou, step)
                self.sw.add_scalar("test/mean_mse", mean_mse, step)
        else:
            pass
        
        loss = loss_sum / cnt
        if self.opt.test_distributed:
            cnt *= self.world_size
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)

        if self.rank == 0:
            speed = cnt / (time.time() - t0)
            print(
                f'rank={self.rank} took {(time.time() - t0):0.2f}s num_images={cnt} speed={speed:.2f}imgs/s loss={loss:0.6f}')
            self.sw.add_scalar("test/loss", loss, step)
        
        self.model.train()
        return loss

    def reconstruct_images(self, masks, reconstructed_images, extra_results_dict, image_src, step,model_save_dir,cnt=0):
        images_for_saving, images_for_logging, mean_iou, mean_biou, mean_mse = make_viz_from_samples(
            masks,
            reconstructed_images,
            extra_results_dict,
            image_src
        )
        root = Path(model_save_dir) / "train_images"
        os.makedirs(root, exist_ok=True)
        for i,img in enumerate(images_for_saving):
            # filename = f"{global_step:08}_s-{i:03}-{fnames[i]}.png"
            filename = f"S{step:04}_{cnt+i:03}_{extra_results_dict['lengths_to_keep'][i]:03}.png"
            path = os.path.join(root, filename)
            img.save(path)

        # 需要可视化长度分布，长度分布的平均值，长度分布的方差，长度分布的熵，长度分布的熵的平均值，长度分布的熵的方差
        return mean_iou, mean_biou, mean_mse