import os
from trainers.helper import set_cuda_visible_devices

set_cuda_visible_devices()

import torch
from trainers.helper import cprint, get_port, get_config
from omegaconf import OmegaConf
import importlib


def dynamic_import(train_script_name):
    module_name = f'trainers.{train_script_name}'
    class_name = 'TrainProcess'

    try:
        module = importlib.import_module(module_name)
        train_class = getattr(module, class_name)
        return train_class
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error importing {class_name} from {module_name}: {e}")
        print(f"Available modules: {importlib.import_module('trainers')}")
        exit()
        return None


def main(rank, local_rank):
    if rank == 0:
        cprint('=> torch version : {}'.format(torch.__version__), 'blue')
        cprint('Initializing Training Process..', 'yellow')

    # Get configuration
    opt = get_config(default_config_path='configs/config_himt.py')
    
    if rank == 0:
        cprint(f"Config:\n{OmegaConf.to_yaml(opt)}", 'green')

    opt.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    opt.gradient_accumulation_steps = max(1, opt.total_batch_size // (opt.world_size * opt.batch_size_per_gpu))

    cprint(f"WORLD_SIZE: {opt.world_size}, RANK: {rank}, LOCAL_RANK: {local_rank}", 'red')

    if rank == 0:
        tensorboard_logpath = os.path.join(opt.model_save_dir, 'logs')
        os.system('rm -r %s/*.*' % tensorboard_logpath)

    train_script_name = opt.train_script_name if hasattr(opt, 'train_script_name') else 'train'
    TrainProcess = dynamic_import(train_script_name)
    p = TrainProcess(rank, local_rank, opt)

    # Save opt as a YAML file
    opt_yaml_path = os.path.join(opt.model_save_dir, 'config.yaml')
    if rank == 0 and not os.path.exists(opt.model_save_dir):
        os.makedirs(opt.model_save_dir)
        with open(opt_yaml_path, 'w') as f:
            OmegaConf.save(config=opt, f=f)

    p.run()


if __name__ == '__main__':
    # os.environ['MASTER_ADDR'] = "localhost"
    # os.environ['MASTER_PORT'] = "8000"
    rank = int(os.environ.get("RANK", '0'))
    local_rank = int(os.environ.get("LOCAL_RANK", '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    ## only use for python debug:
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = "localhost"
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = get_port()

    main(rank, local_rank)
