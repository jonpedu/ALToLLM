import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
from internvl.model.internvl_chat import ALToLLM

# Configurar para usar menos memória
torch.backends.cuda.max_split_size_mb = 512

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Configuração de memória baixa
print("Iniciando carregamento do modelo com configurações de baixa memória...")
path = 'yayafengzi/ALToLLM-8B'

# Usar device_map para distribuir o modelo com offload
model = ALToLLM.from_pretrained(
    path,
    torch_dtype=torch.float16,  # Usar float16 em vez de bfloat16 para economizar memória
    low_cpu_mem_usage=True,
    device_map="auto",  # Usar device mapping automático
    offload_buffers=True  # Usar offload para economizar memória GPU
).eval()

print("Modelo carregado com sucesso!")

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
model.mask_decoder.init_tt_ids(tokenizer)
generation_config = dict(max_new_tokens=512, do_sample=False, return_ids=True)  # Reduzir max_new_tokens

# Processar apenas uma imagem por vez para economizar memória
image_path = './imgs/image1.jpg'
print(f"Carregando imagem: {image_path}")
image = Image.open(image_path).convert('RGB')
pixel_values = load_image(image, max_num=1).to(torch.float16)  # Usar max_num=1 para economizar memória

if torch.cuda.is_available():
    pixel_values = pixel_values.cuda()

num_patches_list = [pixel_values.size(0)]

question = '<image>\nSegment <ref>animal</ref> by adaptive length.'
print("Iniciando inferência...")

with torch.cuda.amp.autocast():
    responses, _, completion_ids_rets = model.batch_chat(tokenizer, pixel_values,
                                num_patches_list=num_patches_list,
                                questions=[question],
                                generation_config=generation_config)

print(f"Resposta: {responses[0]}")

# Processamento do resultado
onehot = torch.zeros(completion_ids_rets.size(0), 32, 1024, dtype=torch.float, device=completion_ids_rets.device)
for i, comp_id in enumerate(completion_ids_rets):
    try:
        start_idx = (comp_id == 92553).nonzero(as_tuple=True)[0][0]
        end_idx = (comp_id == 92554).nonzero(as_tuple=True)[0][0]
        comp_id = comp_id[start_idx + 1:end_idx] - 92555
        valid_len = comp_id.size(0)
        if valid_len > 0:
            onehot[i,:valid_len].scatter_(-1, comp_id.unsqueeze(-1), 1.0)
    except:
        valid_len = 0

image_src = model.convert_image_to_sam_input(pixel_values)*255   
image_src = model.mask_decoder.sam.preprocess(image_src)  
image_embedding = model.mask_decoder.sam.image_encoder(image_src)
masks = model.mask_decoder.decode_prob(onehot,image_embedding=image_embedding).mean(dim=1, keepdim=False).detach()

# Salvar resultado
mask = ((masks[0].float().cpu().numpy()>0.5)*255).astype(np.uint8)
mask = Image.fromarray(mask).resize(image.size)
mask.save('./results/mask_0.png')

print("Máscara salva em './results/mask_0.png'")
print("Inferência concluída com sucesso!")
