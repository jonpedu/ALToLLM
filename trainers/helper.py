import os
import shutil
import random
import torch
import termcolor
import builtins
import subprocess
import pickle
import glob
import re
from omegaconf import OmegaConf
import importlib.util
import json
import yaml


def cprint(*args, **kwargs):
    kwargs['flush'] = True
    # builtins.print(*args, **kwargs)
    # if len(args) > 1:
    #     args = [str(a) for a in args]
    #     args = (''.join(args),)
    termcolor.cprint(*args, **kwargs)


def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)


def dump2File(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def loadFromFile(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def set_cuda_visible_devices():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size < 8:
        rank = int(os.environ.get("RANK", '0'))
        os.environ['CUDA_VISIBLE_DEVICES'] = get_gpu_id(world_size, verbose=rank==0)


def getFolderList(directory, join=False):
    folders = os.listdir(directory)

    folders_match = []
    for fd in folders:
        fd_join = os.path.join(directory, fd)
        if os.path.isdir(fd_join):
            if join:
                folders_match.append(fd_join)
            else:
                folders_match.append(fd)
    return folders_match


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_module_from_file(name, path):
    import importlib
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_opt_from_python_config(config_file):
    X = load_module_from_file('', config_file)

    config_keys = [k for k in dir(X) if not k.startswith('_')]
    config = {k: getattr(X, k) for k in config_keys}
    opt = AttrDict(config)
    return opt


# def build_env(config, config_name, ckpt_path):
#     t_path = os.path.join(ckpt_path, config_name)
#     if config != t_path:
#         os.makedirs(ckpt_path, exist_ok=True)
#         shutil.copyfile(config, os.path.join(ckpt_path, config_name))


def init_seeds(seed=0):
    # seed = 0
    # sets the seed for generating random numbers.
    torch.manual_seed(seed)
    # Sets the seed for generating random numbers for the current GPU.
    # It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed(seed)
    # Sets the seed for generating random numbers on all GPUs.
    # It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    if seed == 0:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_port():
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    used_ports = subprocess.getoutput(pscmd).split()
    port = str(random.randint(15000, 20000))
    if port not in used_ports:
        return port
    else:
        get_port()


def add_date_prefix_to_tag(tag):
    from datetime import datetime
    import pytz
    # Get the current time in the timezone +8
    timezone = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(timezone)
    # Format the current time
    formatted_time = current_time.strftime("%y%m%d.%H%M%S")

    return formatted_time + '_' + tag


def get_lastest_ckpt(ckpt_path, verbose=False):
    if not os.path.isdir(ckpt_path):
        return ckpt_path

    files = glob.glob(os.path.join(ckpt_path, '*.pth'))
    if len(files) == 0:
        return None

    timestamps = [(i, os.path.getmtime(file)) for i, file in enumerate(files)]
    timestamps.sort(reverse=True, key=lambda x: x[1])
    if verbose:
        for i, _ in timestamps:
            print(files[i])
    return files[timestamps[0][0]]


def get_best_ckpt(ckpt_dir, verbose=False):
    if not os.path.isdir(ckpt_dir):
        return ckpt_dir
    # 匹配模型文件名中的损失值
    pattern = re.compile(r'_L(\d+\.\d+)')
    min_loss = float('inf')
    best_ckpt = None

    # 遍历目录中的所有文件
    for filename in os.listdir(ckpt_dir):
        if filename.endswith('.pth'):
            # 查找损失值
            match = pattern.search(filename)
            if match:
                # 转换损失值为浮点数并比较
                loss = float(match.group(1))
                if loss < min_loss:
                    min_loss = loss
                    best_ckpt = filename

    # 如果找到了最佳检查点
    if best_ckpt:
        best_ckpt_path = os.path.join(ckpt_dir, best_ckpt)
        if verbose:
            print(f'Best checkpoint: {best_ckpt_path} with loss: {min_loss}')
        return best_ckpt_path
    else:
        if verbose:
            print('No checkpoint found.')
        return None


############################## get_gpu_id ##############################
def get_gpu_info(visible_gpus=None):
    res = subprocess.getoutput('nvidia-smi')

    res = res.split('\n')

    gpu_info = []
    for i, s in enumerate(res):
        flag = True
        for x in ['%', 'W', 'C', 'MiB']:
            flag = flag and (x in s)
        if not flag:
            continue

        id = len(gpu_info)
        info = s.split(' ')
        # print(info)

        pwr_use, pwr_total = [float(x.split('W')[0]) for x in info if 'W' in x]
        mem_use, mem_total = [float(x.split('MiB')[0]) for x in info if 'MiB' in x]

        if visible_gpus is None or int(id) in visible_gpus:
            gpu_info.append({'id': id, 'mem_use': mem_use, 'mem_total': mem_total,
                             'pwr_use': pwr_use, 'pwr_total': pwr_total})
    return gpu_info


def get_gpu_id(num_gpus=1, max_use_mem=1000000, visible_gpus=None, verbose=True, reverse=True):
    res = subprocess.getoutput('nvidia-smi')

    res = res.split('\n')

    gpu_info = []
    for i, s in enumerate(res):
        flag = True
        for x in ['%', 'W', 'C', 'MiB']:
            flag = flag and (x in s)
        if not flag:
            continue

        id = len(gpu_info)
        info = s.split(' ')
        # print(info)

        pwr_use, pwr_total = [float(x.split('W')[0]) for x in info if 'W' in x]
        mem_use, mem_total = [float(x.split('MiB')[0]) for x in info if 'MiB' in x]

        if visible_gpus is None or int(id) in visible_gpus:
            gpu_info.append({'id': id, 'mem_use': mem_use, 'mem_total': mem_total,
                             'pwr_use': pwr_use, 'pwr_total': pwr_total})

    is_busy = lambda gpu_info: gpu_info['mem_use'] <= max_use_mem

    def get_valid_ids(gpu_info):
        gpu_info.sort(key=lambda x: x['mem_use'], reverse=False)
        if gpu_info[0]['mem_use'] > 1000:
            print('All gpu is busy!')

        # podA = [x for x in gpu_info[:3]]
        # podB = [x for x in gpu_info[4:]]
        # is_busy(gpu_info[0])

        ids = []
        ids_str = ''
        for i in range(min(num_gpus, len(gpu_info))):
            if gpu_info[i]['mem_use'] <= max_use_mem:
                ids.append(gpu_info[i]['id'])
                if len(ids_str) > 0:
                    ids_str += ','
                ids_str += str(gpu_info[i]['id'])

        return ids, ids_str

    if reverse:
        ids, ids_str = get_valid_ids(gpu_info[::-1].copy())
    else:
        ids, ids_str = get_valid_ids(gpu_info.copy())

    if verbose:
        print('********************* GPU INFO *********************')
        for x in gpu_info:
            if x['id'] in ids:
                print(' √', end='')
            else:
                print('  ', end='')

            print(' GPU_id:%d   Mem:%5.0fMB/%0.0fMB   Power:%3.0fW/%0.0fW' %
                  (x['id'], x['mem_use'], x['mem_total'], x['pwr_use'], x['pwr_total']))

        print('********************* GPU INFO *********************')

    return ids_str


############################## load_config ##############################
def load_json_config(file_path):
    """Load config from a JSON file"""
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def load_yaml_config(file_path):
    """Load config from a YAML file"""
    return OmegaConf.load(file_path)

def load_python_config(file_path):
    """Load config from a Python file into a dictionary"""
    spec = importlib.util.spec_from_file_location("config", file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Get all non-private attributes from the module
    config_dict = {k: v for k, v in config_module.__dict__.items() 
                  if not k.startswith('_')}
    return config_dict

def get_config(default_config_path='configs/config_debug.py'):
    """
    Get configuration from config file and CLI arguments.
    Supports .py, .yaml, .yml, and .json config files.
    Priority: CLI args > config file
    
    Args:
        default_config_path: Default path to config file if not specified in CLI
    
    Returns:
        OmegaConf configuration object
    """
    # Get CLI arguments
    cli_conf = OmegaConf.from_cli()
    
    # Get config file path
    config_path = cli_conf.get('config', default_config_path)
    
    # Determine file type and load accordingly
    file_ext = os.path.splitext(config_path)[1].lower()
    
    if file_ext == '.py':
        base_conf = load_python_config(config_path)
    elif file_ext in ['.yaml', '.yml']:
        base_conf = load_yaml_config(config_path)
    elif file_ext == '.json':
        base_conf = load_json_config(config_path)
    else:
        raise ValueError(f"Unsupported config file type: {file_ext}")
    
    # Convert to OmegaConf if it's not already
    if not isinstance(base_conf, OmegaConf):
        base_conf = OmegaConf.create(base_conf)
    
    # Merge configurations (CLI overrides config file)
    conf = OmegaConf.merge(base_conf, cli_conf)
    
    # Post-process some configs
    if conf.get('total_batch_size') is None:
        conf.total_batch_size = conf.batch_size_per_gpu * conf.num_gpus
    
    # Create model save directory
    conf.model_save_dir = os.path.join(conf.run_dir, conf.tag)
    
    return conf
