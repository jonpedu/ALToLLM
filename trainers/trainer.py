import os
import time
import numpy as np
import math
import torch
from termcolor import cprint
import glob
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, mom=0):
        self.mom = mom
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if torch.is_tensor(val):
            if val.is_cuda:
                val = val.data.float().cpu().numpy()
            else:
                val = val.data.float().numpy()
        if isinstance(val, np.ndarray):
            val = val.item()

        if self.count == 0:
            self.val = val
        else:
            self.val = self.mom * self.val + (1 - self.mom) * val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


# function to calculate the Exponential moving averages for the Generator weights
# This function updates the exponential average weights based on the current training
@torch.no_grad()
def update_average(model_tgt, model_src, beta=0.9):
    """
    update the model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates the target model)
    """

    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    # toggle_grad(model_tgt, False)
    # toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)
        # p_tgt.data.copy_(beta * p_tgt.data + (1. - beta) * p_src.data)

    # turn back on the gradient calculation
    # toggle_grad(model_tgt, True)
    # toggle_grad(model_src, True)


class WarmupScheduler:
    """Warmup wrapper for learning rate scheduler"""
    def __init__(self, optimizer, warm_steps, scheduler=None):
        self.optimizer = optimizer
        self.warm_steps = warm_steps
        self.scheduler = scheduler
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step < self.warm_steps:
            # Linear warmup
            lr_scale = self.current_step / self.warm_steps
            for pg in self.optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * lr_scale
        elif self.scheduler is not None:
            self.scheduler.step()
            
        return self.optimizer.param_groups[0]['lr']


class TrainerBase(object):
    """
    TrainerBase
    """

    def __init__(self, FLAGS=None):
        pass

    @staticmethod
    def set_dtype(dtype_str):
        dtype_map = {
            'float32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16
        }
        return dtype_map.get(dtype_str.lower(), torch.float32)

    def set_meter(self, names):
        for name in names:
            name = 'meter_' + name
            if not hasattr(self, name):
                self.__setattr__(name, AverageMeter(0))
        return

    def update_meter(self, loss):
        for k, v in loss.items():
            name = 'meter_' + k
            if hasattr(self, name):
                self.__getattribute__(name).update(v)
        return

    def reset_meter(self):
        for name in self.__dict__.keys():
            if name.startswith('meter_'):
                self.__getattribute__(name).reset()
        return

    def get_meter(self, str=False):
        m = {}
        for name in self.__dict__.keys():
            if name.startswith('meter_'):
                v = self.__getattribute__(name).avg
                nm = name.split('meter_')[-1]
                m[nm] = v
        if str:
            s = ''
            for nm, v in m.items():
                if nm == 'newline':
                    s += '\n          '
                elif abs(v) > 1e-2:
                    s += ' %s=%0.4f' % (nm, v)
                elif abs(v) > 1e-3:
                    s += ' %s=%0.5f' % (nm, v)
                elif abs(v) > 1e-4:
                    s += ' %s=%0.6f' % (nm, v)
                elif v == 0:
                    s += ' %s=%0.0f' % (nm, v)
                else:
                    s += ' %s=%0.3e' % (nm, v)
            return s
        else:
            return m

    def get_lastest_ckpt(self, dir, print=False):
        if not os.path.isdir(dir):
            return None

        files = glob.glob(os.path.join(dir, '*.pth'))
        if len(files) == 0:
            return None

        timestamps = [(i, os.path.getmtime(file)) for i, file in enumerate(files)]
        timestamps.sort(reverse=True, key=lambda x: x[1])
        if print:
            for i, _ in timestamps:
                print(files[i])
        return files[timestamps[0][0]]

    def delete_older_ckpt_dir(self, dir, maxN=2, verbose=True):
        folders = glob.glob(dir)
        if len(folders) <= maxN:
            return None

        timestamps = [(f, os.path.getmtime(f)) for i, f in enumerate(folders) if os.path.isdir(f)]
        timestamps.sort(reverse=True, key=lambda x: x[1])

        for ts in timestamps[maxN:]:
            cmd = 'rm -r %s' % ts[0]
            if verbose:
                print(cmd)
            os.system(cmd)
        return None

    def delete_older_ckpt(self, dir, maxN=2, verbose=True):
        # return if rank is not 0
        if self.rank != 0:
            return None

        if not os.path.isdir(dir):
            return None

        files = glob.glob(os.path.join(dir, '*.pth'))
        if len(files) <= maxN:
            return None

        timestamps = [(file, os.path.getmtime(file)) for i, file in enumerate(files)]
        timestamps.sort(reverse=True, key=lambda x: x[1])

        for ts in timestamps[maxN:]:
            cmd = 'rm %s' % ts[0]
            if verbose:
                print(f"[delete_older_ckpt] rank={self.rank} max_save={maxN} cmd={cmd}")
            os.system(cmd)
        return None

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def adjust_learning_rate(self, type, step, optimizer, **config):
        raise NotImplementedError

    def set_param_lr(self, optims, lr):
        if isinstance(optims, list):
            for optim in optims:
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
        else:
            for param_group in optims.param_groups:
                param_group['lr'] = lr
        return None

    def adjust_learning_rate_sqrt(self, step, optimizer, **config):
        max_lr = config.get('max_lr', None) or config.get('learning_rate', None)
        min_lr = config.get('min_lr', 1e-6)
        warm_steps = config.get('warm_steps', 0)
        init_steps = config.get('init_steps', 10000)

        if step < warm_steps:
            scale = step / warm_steps
        else:
            scale = (init_steps / (step - warm_steps + init_steps)) ** 0.5

        lr = max_lr * scale
        lr = max(lr, min_lr)
        self.set_param_lr(optimizer, lr)
        self.current_lr = lr
        return lr

    def adjust_learning_rate_cosine(self, step, optimizer, **config):
        start_lr = 0.000001
        max_lr = config.get('max_lr', None) or config.get('learning_rate', None)
        min_lr = config.get('min_lr', 1e-6)
        warm_steps = config.get('warm_steps', 0)
        total_steps = config['total_steps']

        if step >= total_steps and step < total_steps + 20:
            print('STOP TRAIN!!! step=%d lr=%0.6f' % (step, self.current_lr))

        if step < warm_steps:
            lr = ((max_lr - start_lr) * step) / warm_steps + start_lr
        else:
            step = min(step, total_steps - 1e-6)
            lr = max_lr * (math.cos(math.pi * (step - warm_steps) / (total_steps - warm_steps)) + 1) / 2

        lr = max(lr, min_lr)
        self.set_param_lr(optimizer, lr)
        self.current_lr = lr
        return lr

    def adjust_learning_rate_exponential(self, epoch, step, optimizer, **config):
        learning_rate = config['learning_rate']
        warm_steps = config.get('warm_steps', 0)
        lr_decay = config.get('lr_decay', 0)
        if step < warm_steps:
            scale = step / warm_steps
        else:
            scale = 1.0

        scale *= lr_decay ** epoch
        lr = learning_rate * scale

        self.set_param_lr(optimizer, lr)
        self.current_lr = lr
        return lr

    def save_model(self, epoch, step, loss,
                   save_dir,
                   higher_is_better=False,
                   max_time_not_save=60 * 60,
                   save_ema=True):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not hasattr(self, 'best'):
            is_best = True
            self.best = loss
        else:
            is_best = loss > self.best if higher_is_better else loss < self.best

        long_time_save = True
        if hasattr(self, 'last_save_timestamp'):
            dt = time.time() - self.last_save_timestamp
            if max_time_not_save is None or dt < max_time_not_save:
                long_time_save = False

        if is_best or long_time_save:
            self.best = max(loss, self.best) if higher_is_better else min(loss, self.best)

            model_name = 'Model_E{}S{}_L{:.6f}.pth'.format(epoch, step, loss)
            model_path = os.path.join(save_dir, model_name)
            cprint('save model to >>> %s' % model_path, 'cyan')
            state = {'epoch': epoch,
                     'step': step,
                     'best': self.best,
                     'model': self.model.state_dict(),
                     'optimizer': self.optim.state_dict(),
                     'config': self.opt.__dict__
                     }
            if hasattr(self, 'model_ema') and save_ema:
                state['model_ema'] = self.model_ema.state_dict()
            torch.save(state, model_path)
            self.last_save_timestamp = time.time()

            # Delete older checkpoints if exceeding max number
            if hasattr(self.opt, 'num_max_save_models'):
                self.delete_older_ckpt(save_dir, 
                                     maxN=self.opt.num_max_save_models, 
                                     verbose=True)
        return False

    def get_match_ckpt(self, model, ckpt_src):
        if model is None: return None
        ckpt = model.state_dict()
        for k, v in ckpt.items():
            k_without_module = k[7:] if 'module.' in k else k
            k_module = 'module.' + k_without_module
            if k_module in ckpt_src and ckpt[k].shape == ckpt_src[k_module].shape:
                ckpt[k] = ckpt_src[k_module].to(v.device)
            elif k_without_module in ckpt_src and ckpt[k].shape == ckpt_src[k_without_module].shape:
                ckpt[k] = ckpt_src[k_without_module].to(v.device)
            else:
                if self.rank == 0:
                    print('%s is not loaded.' % k)
        return ckpt

    def add_model_graph(self, model, input_shape):
        """Add model graph to tensorboard"""
        if hasattr(self, 'sw'):
            dummy_input = torch.randn(input_shape).to(self.device)
            self.sw.add_graph(model, dummy_input)
    
    def log_memory_stats(self, step):
        """Log GPU memory usage"""
        if hasattr(self, 'sw') and self.rank == 0:
            cuda_memory_total = torch.cuda.get_device_properties(self.device).total_memory
            cuda_memory_allocated = torch.cuda.memory_allocated(self.device)
            cuda_memory_cached = torch.cuda.memory_reserved(self.device)
            self.sw.add_scalar('system/gpu_memory_allocated', 
                             cuda_memory_allocated, 
                             step)
            self.sw.add_scalar('system/gpu_memory_used%',
                             cuda_memory_allocated/cuda_memory_total*100,
                             step)

    def create_scheduler(self, optimizer, schedule_type='cosine', **kwargs):
        """Create learning rate scheduler with optional warmup"""
        warm_steps = kwargs.get('warm_steps', 0)
        total_steps = kwargs.get('total_steps')
        min_lr = kwargs.get('min_lr', 1e-6)
        max_lr = kwargs.get('max_lr', optimizer.param_groups[0]['lr'])
        
        # Set initial learning rate for warmup
        for pg in optimizer.param_groups:
            pg['initial_lr'] = max_lr
        
        if schedule_type == 'cosine':
            # Custom cosine schedule that maintains min_lr after total_steps
            def cosine_decay(step):
                if step < warm_steps:
                    return step / warm_steps
                elif step >= total_steps:
                    return min_lr / max_lr
                else:
                    progress = (step - warm_steps) / (total_steps - warm_steps)
                    return min_lr / max_lr + 0.5 * (1 - min_lr / max_lr) * (1 + math.cos(math.pi * progress))
                
            scheduler = LambdaLR(optimizer, cosine_decay)
        elif schedule_type == 'sqrt':
            # Custom sqrt decay
            def sqrt_decay(step):
                if step < warm_steps:
                    return step / warm_steps
                init_steps = kwargs.get('init_steps', 10000)
                return max(
                    min_lr / max_lr,
                    (init_steps / (step - warm_steps + init_steps)) ** 0.5
                )
            scheduler = LambdaLR(optimizer, sqrt_decay)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        return WarmupScheduler(optimizer, warm_steps, scheduler)
    

    def verify_weights_sync(self):
        """Verify that weights are synchronized across GPUs"""
        if self.opt.world_size > 1:
            for name, param in self.model.named_parameters():
                if torch.is_tensor(param):
                    # Sum the parameter tensor across all processes
                    param_sum = param.sum()
                    torch.distributed.all_reduce(param_sum)
                    
                    # If parameter sum is different when divided by world_size,
                    # weights are not in sync
                    if self.rank == 0:
                        if not torch.allclose(param_sum / self.opt.world_size, param.sum()):
                            print(f"Warning: Parameter {name} is not synchronized across GPUs")

    def load_ckpt_if_exist(self, model_path, verbose=True):
        if not model_path:
            return

        # Check file extension
        is_safetensor = model_path.endswith('.safetensors')
        
        if is_safetensor:
            try:
                from safetensors.torch import load_file
                model_weight = load_file(model_path)
                model_weight['quantize.embedding.weight'] = model_weight['quantize.embedding.weight'][:1024,...]
                # delete the key with the name pixel_decoder.conv_in.weight
                del model_weight['pixel_decoder.conv_in.weight']
            except ImportError:
                if self.rank == 0:
                    print("Please install safetensors: pip install safetensors")
                return
        else:
            model_info = torch.load(model_path, map_location="cpu", weights_only=False)
            if 'model' in model_info:    
                model_weight = model_info['model']
            else:
                model_weight = model_info

        # Remove 'module.' prefix if present
        model_weight = {k.replace('module.', ''): v for k, v in model_weight.items()}
        
        # Get the actual model (remove DDP wrapper if present)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Handle shape mismatches between checkpoint and model weights
        model_state_dict = model.state_dict()
        for key in list(model_weight.keys()):
            if key in model_state_dict:
                # Check if shapes don't match
                if model_weight[key].shape != model_state_dict[key].shape:
                    if self.rank == 0 and verbose:
                        print(f"Shape mismatch for {key}: checkpoint {model_weight[key].shape} vs model {model_state_dict[key].shape}")
                    
                    # Handle different cases of shape mismatch
                    if len(model_weight[key].shape) == len(model_state_dict[key].shape):
                        try:
                            # Check if checkpoint weights are larger in any dimension
                            is_larger = any(c > m for c, m in zip(model_weight[key].shape, model_state_dict[key].shape))
                            is_smaller = any(c < m for c, m in zip(model_weight[key].shape, model_state_dict[key].shape))
                            
                            if is_larger and not is_smaller:
                                # Create a slice for each dimension
                                slices = []
                                for dim_size, target_size in zip(model_weight[key].shape, model_state_dict[key].shape):
                                    if dim_size > target_size:
                                        slices.append(slice(0, target_size))
                                    else:
                                        slices.append(slice(None))
                                
                                # Apply the slices to crop the tensor
                                model_weight[key] = model_weight[key][tuple(slices)]
                                if self.rank == 0 and verbose:
                                    print(f"  Successfully cropped {key} from {model_weight[key].shape} to {model_state_dict[key].shape}")
                            else:
                                # If checkpoint weights are smaller in any dimension, we skip this parameter
                                if self.rank == 0 and verbose:
                                    print(f"  Cannot adapt {key} - checkpoint weights will be skipped for this parameter")
                        except Exception as e:
                            if self.rank == 0 and verbose:
                                print(f"  Error adapting {key}: {e}")
        
        # model_weight['encoder.latent_token_positional_embedding2'] = model_weight['encoder.latent_token_positional_embedding']
        msg = model.load_state_dict(model_weight, strict=False)
        
        if self.rank==0 and verbose:
            print(f"\nLoading weights from: {model_path}")
            print("\nWeight loading details:")
            print(f"{'='*80}")
            if len(msg.missing_keys) > 0:
                print("Missing keys:")
                for key in msg.missing_keys:
                    print(f"  - {key}")
            if len(msg.unexpected_keys) > 0:
                print("\nUnexpected keys:")
                for key in msg.unexpected_keys:
                    print(f"  - {key}")
            if len(msg.missing_keys) == 0 and len(msg.unexpected_keys) == 0:
                print("All keys matched successfully!")
            print(f"{'='*80}\n")

        # Only load optimizer state if using .pth format and optimizer state exists
        if not is_safetensor and 'optim' in model_info:
            model_optim = model_info['optim']
            model_optim = {k.replace('module.', ''): v for k, v in model_optim.items()}
            self.optim.load_state_dict(model_optim)
        

class DistributedTrainerBase(torch.multiprocessing.Process, TrainerBase):
    """
    DistributedTrainerBase
    """

    def __init__(self, ):
        super().__init__()
        pass

    def count_parameter(self, model, tag=''):
        num_total, num_learn = 0, 0
        for p in model.parameters():
            num_total += p.numel()
            num_learn += p.numel() if p.requires_grad else 0
        cprint(f'{tag} learn parameters: {(num_learn):,}   total parameters: {(num_total):,}', color='red')

    def test_dataloader_speed(self, dataloader):
        cnt = 0
        t0 = time.perf_counter()
        for i, mb in enumerate(dataloader):
            keyname = list(mb.keys())[0]
            cnt += len(mb[keyname])
            if time.perf_counter() - t0 > 2 or i >= 3:
                dt = time.perf_counter() - t0
                speed = cnt / dt
                print(f"[dataloader_speed] rank={self.rank} count={cnt}  dt={dt:0.2f}s speed={speed:0.2f}:samples/s ")
                cnt = 0
                t0 = time.perf_counter()
            if i >= 3:
                break
        return speed
