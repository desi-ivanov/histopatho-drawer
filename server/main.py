from torchvision import transforms
import torch
from flask import Flask, request, jsonify, make_response
import pickle
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from functools import partial
import pickle
import numpy as np
import dnnlib
from torch_utils import misc
import training.networks
from training.dataset_vae import ImageFolderDataset, ZipDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

training_set_seg = ImageFolder("./data/segmented/train", transform=transforms.ToTensor(), loader=Image.open)
training_set_imgs = ImageFolder("./data/real/train", transform=transforms.ToTensor(), loader=Image.open)

full_datset = ZipDataset(training_set_seg, training_set_imgs)

def load_vae(resume_pkl):

    common_kwargs = dict(c_dim=0, img_resolution=512, img_channels=3)
    VAE =  training.networks.AEStyleGenerator(
        w_dim = 512,
        z_dim = 512,
        mapping_kwargs ={
            "num_layers": 8
        },
        synthesis_kwargs = {
            "conv_clamp": 256,
            "channel_max": 512,
            "channel_base": 32768,
            "num_fp16_res": 4
        },
        **common_kwargs,
    ).requires_grad_(False).cpu()
    with dnnlib.util.open_url(resume_pkl) as f:
        resume_data = pickle.load(f)
    misc.copy_params_and_buffers(resume_data['VAE'], VAE, require_all=False)
    return partial(VAE, c=None, force_fp32=True)
print(training_set_seg[0][0][0].shape)
app = Flask(__name__)
model = load_vae("./network-snapshot-001196.pkl")

def torch_to_b64(tensor):
    image = Image.fromarray(tensor.permute(1,2,0).cpu().detach().numpy().astype(np.uint8))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    result_str = base64.b64encode(buffered.getvalue())
    result_str ='data:image/PNG;base64,' + result_str.decode('utf-8')
    return result_str

@app.route("/infer", methods=['POST', 'OPTIONS'])
def infer():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "POST":
        content = request.json
        in_img = content['img'].split(',')[-1]
        in_img = Image.open(BytesIO(base64.b64decode(in_img + "=" * ((4 - len(in_img) % 4) % 4))))
        in_img = transforms.ToTensor()(in_img).unsqueeze(0).cpu().mul(2).sub(1)
        print(in_img.unique())
        #  remove alpha channel
        in_img = in_img[:, :3, :, :]
        image = model(in_img)[0]
        return _corsify_actual_response(jsonify({
            'img': torch_to_b64(image.add(1).div(2).mul(255).clip(0, 255))
        }))
        

@app.route("/example", methods=['GET', 'OPTIONS'])
def example():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "GET":
        (seg_img, _c), (real_img, _c) = full_datset[np.random.randint(0, len(full_datset))]
        seg_img = seg_img.div(255).mul(2).sub(1)
        image = model(seg_img.unsqueeze(0).cpu())[0].add(1).div(2).mul(255).clip(0, 255)
        print(seg_img.unique())
        return _corsify_actual_response(jsonify({
            'segmented': torch_to_b64(seg_img.add(1).div(2).mul(255).clip(0, 255)), 
            'reconstructed': torch_to_b64(image),
            'real': torch_to_b64(real_img)
        }))

def _build_cors_preflight_response():
    response = make_response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response