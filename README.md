# Ultrasound Image Generation using Latent Diffusion Models

Code used for following paper: [...]

Datasets: [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset), [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/)


Forked SD: https://github.com/Toterino/stable-diffusion

Forked CN: https://github.com/Toterino/ControlNet

> **_NOTE:_** the hyperparameters / paths used in the code below are slightly different than the forked repositories; it is indicated clearly

# Stable Diffusion
The Official Stable Diffusion [repository](https://github.com/CompVis/stable-diffusion) was forked and used in conjunction with Justin Pinkney's (a ML researcher) [repository](https://github.com/justinpinkney/stable-diffusion) and their [guide](https://github.com/LambdaLabsML/examples/tree/main/stable-diffusion-finetuning) on finetuning. 

Since I was having trouble finetuning on the original repository and I wanted to understand what made Pinkney's code work, I slowly added a few pieces of their code to the official code, slightly tweaked them and made it work. The entire BUSI dataset was used.

> **_NOTE:_** the following guide is thus written ONLY for reproducibility purposes on BUSI; the recommended and proper way to finetune stable diffusion is to directly use Justin Pinkney's repository and to follow their corresponding guide.
> 
### _root_dir_
Stable Diffusion 1.5 was downloaded from the hugging face [page](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) (v1-5-pruned.ckpt) and placed in ```models/ldm/sdv1-5/```
The kl-f8 autoencoder was downloaded and placed in the correct folder.

The following terminal command was executed to install all prerequisities: ```conda create environment.yaml```


### _simple.py_ 
This file is used from Pinkney's repository; it facilitates the setup of the local dataset. The HF class was removed which is not a necessary change to make. The ext attribute in the FolderData class was changed to 'png'

### _train.yaml_
The ```pokemon.yaml``` file from Pinkey's repository was used and slightly modified. 

The learning rate was changed. For BUSI, 1.0e-06 was used but --scale_lr was True (became 2.0e-06); it is equivalent to making the base_learning_rate 2.0e-06 and setting --scale_lr to False
```python
base_learning_rate: 1.0e-06
```
The target class was changed to accomodate a local dataset. The dataset images and jsonl prompt files (provided in the fork, but should be placed elsewhere) were set up in a specific path:

```python
    batch_size: 2
    num_workers: 2
    train:
      target: ldm.data.simple.FolderData
      params:
        root_dir: ../data/BUSI_DATA/images # path to the dataset
        caption_file: ../data/BUSI_DATA/prompts.jsonl # prompts 
```

The following change was made only for BUSI (removed for CAMUS). This should work (not sure if its with the "" or not), but if there is a problem with the config file, try removing this section:
```python
validation:
      target: ldm.data.simple.TextOnly
      params:
        captions:
        - "ultrasound image"
        output_size: 512
        n_gpus: 2
```
This change is done to save only the weights:
```python
  modelcheckpoint:
    params:
      every_n_train_steps: 10000 # Modified to our needs
      save_top_k: -1
      monitor: null
      save_weights_only: true # Only save the weights of checkpoint, this wasn't done for BUSI
```
The following line was commented out order to avoid bugs:
```python
#log_all_val: True
```

### _ddpm.py_
The 'txt' from Pinkey's repository was added as it is necessary for the text prompts:
```python
if cond_key in ['caption', 'coordinates_bbox','txt']: # line taken from https://github.com/justinpinkney/stable-diffusion/blob/main/ldm/models/diffusion/ddpm.py
```

### _main.py_

The following lines make 1 GPU available:
```python
# Code to change what GPUs are available
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print(torch.cuda.device_count())
```
The following lines from Pinkey's repository were used to load the model state for finetuning:
```python
parser.add_argument( # taken from https://github.com/justinpinkney/stable-diffusion/blob/main/main.py
        "--finetune_from",
        type=str,
        nargs="?",
        default="",
        help="path to checkpoint to load model state from"
    )
```
```python
# Taken from https://github.com/justinpinkney/stable-diffusion/blob/main/main.py
if not opt.finetune_from == "":  
            print(f"Attempting to load state from {opt.finetune_from}")
            old_state = torch.load(opt.finetune_from)
            if "state_dict" in old_state:
                print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
                old_state = old_state["state_dict"]

            m, u = model.load_state_dict(old_state, strict=False)
            if len(m) > 0:
                print("missing keys:")
                print(m)
            if len(u) > 0:
                print("unexpected keys:")
                print(u)

```

### _root_dir_

Other slight modifications were done for the following reasons:
- to fix a bug in ```main.py```.
- to make the sampling time more obvious in ```txt2img.py```.
- to check that the image prompt association is correct in ```ddpm.py```.

## Training

The following terminal command was executed:

```python
CUDA_VISIBLE_DEVICES=3 python main.py --base configs/stable-diffusion/train.yaml --train --gpus=1 --finetune_from models/ldm/sdv1-5/v1-5-pruned.ckpt
```
```--scale_lr False``` can be used to make the lr equal to the base one defined in the config file. It was not used for the BUSI model and so the lr for BUSI became 2e-06

``` --seed 321 ``` can be used to set the seed number for training. The default one was used for BUSI which is 23.
The models are saved in the ```logs``` folder.

## Sampling

The following terminal command was executed:

```python
python scripts/txt2img.py --prompt "Ultrasound image of benign breast" --ckpt logs/FS_BUSI/checkpoints/epoch=000051.ckpt --scale 7 --plms --seed 1 --n_sample 10 --n_iter 10
```
```--n_sample``` represents the number of columns.

```--n_iter``` represents the number of rows.

```--plms ``` represents the sampler that was used.

All samples were generated using a guidance scale of 7 with seed numbers 1 and 2.
__________________________________________________________________________________________________________________________________________________________________________________

# ControlNet

The official [repository](https://github.com/lllyasviel/ControlNet) for ControlNet was forked and has a [guide](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md) on how to train on your own data. The setup will be explained ONLY for reproducibility purposes on BUSI.

> **_NOTE:_** the following dataset images were not used for training, only for inferencing:

Benign: [ 403, 437 ] 

Malignant:  [ 186, 210 ] 

Normal: [ 114, 133 ]  

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
> **_NOTE:_** the following masks were superimposed:

Benign: 4, 25, 54, 58, 83, 92, 93, 98, 100, 163, 173, 181, 195, 315, 346

Malignant: 53

```python
import cv2

im1 = cv2.imread("./Dataset_BUSI_with_GT/malignant/malignant (53)_mask.png")
im2 = cv2.imread("./Dataset_BUSI_with_GT/malignant/malignant (53)_mask_1.png")

finalim = im1 + im2
cv2.imwrite('./fixed_masks/malignant (53)_mask.png', finalim) # https://stackoverflow.com/questions/69996609/how-to-save-png-images-with-opencv
```

### _root_dir_
The following folder structure was made and the data was inserted in the corresponding folders (source is condition, target is gt):
```
training/BUSI/source/
training/BUSI/target/
```
A prompts.json file was created in ```training/BUSI``` (the project json files are provided):

### _tutorial_dataset.py_
The following lines were adjusted to include the correct path:
  
```python 
with open('./training/BUSI/prompt_busi.json', 'rt') as f:
```

```python 
source = cv2.imread('./training/BUSI/' + source_filename) # "source" is the condition
```

```python 
target = cv2.imread('./training/BUSI/' + target_filename) # "target" is the data
```

The dataset images were resized to the same shape (512x512 in our case):
```python 
source = cv2.resize(source, (512,512), interpolation= cv2.INTER_LINEAR) # https://learnopencv.com/image-resizing-with-opencv/#resize-by-wdith-height
target = cv2.resize(target, (512,512), interpolation= cv2.INTER_LINEAR) # https://learnopencv.com/image-resizing-with-opencv/#resize-by-wdith-height
```

### _root_dir_

A model from SD that was finetuned was placed in ```./models/```

The following terminal command, which initializes a CN model, was executed:
```python
python tool_add_control.py ./models/YourModelFromSD.ckpt ./models/old_busi_control_sd15_ini.ckpt
```

### _logger.py_

A part of line 43 was uncommented in order to log images only after a certain number of steps and not after every epoch.


# Training

### _tutorial_train.py_
The hyperparameters and code were adjusted for BUSI:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Use only GPU #3

resume_path = './models/old_busi_control_sd15_ini.ckpt' # We changed the path accordingly
batch_size = 4 
logger_freq = 5000 # We changed the logging frequency
learning_rate = 1e-5 
sd_locked = True 
only_mid_control = False
```
All checkpoints are saved only after a certain number of steps:
```python
# Making a checkpoint callback that saves every model after a certain number of steps
call_checkpoint = ModelCheckpoint( # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
    every_n_train_steps = 5000,  # Every 5000 steps
    save_top_k=-1  # All checkpoints are saved
)

dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, call_checkpoint]) # You have to add call_checkpoint here
```
The following terminal command was executed and the models were saved in the ```lightninglogs``` folder
```python
CUDA_VISIBLE_DEVICES=3 python tutorial_train.py
```


# Sampling

### _sampling.py_
The code was mostly taken from gradio_seg2image.py

The path on line 89 was adjusted (version 4 is BUSI):
```python
model.load_state_dict(load_state_dict('./lightning_logs/version_4/checkpoints/epoch=85-step=14999.ckpt', location='cuda'))
```
The code was adjusted in the following way to obtain our results:
```python
prompt = "Ultrasound image of normal breast" # Change the prompt here 
guidance_scale = 5 # You can play with the guidance scale, we kept it at 5
samples_row = 4 # how many samples per row, this is just for the matplotlib grid

# Change the condition here
source_1 = cv2.resize(cv2.imread("./inference/BUSI/masks/benign (412)_mask.png"), (512,512), interpolation= cv2.INTER_LINEAR) # https://learnopencv.com/image-resizing-with-opencv/
# Change the ground truth here
target_1 = cv2.resize(cv2.imread("./inference/BUSI/gt/benign (412).png"), (512,512), interpolation= cv2.INTER_LINEAR)

sample_img(source_1, prompt, "", "", 8, 50, False, guidance_scale, 1, samples_row)
# 8 is the number of samples you want to generate
# 50 is the number of DDIM steps
# 1 is seed number
# I made strength = 1 in the "sample_img" function, since that is its default value in gradio_seg2image.py
```
The following terminal command was executed:
```python
CUDA_VISIBLE_DEVICES=3 python sampling.py
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Preparing the CAMUS Dataset:
```python
img.append(nibabel.load(dir + "/" + files[ind[i]]).get_fdata()) #https://neuraldatascience.io/8-mri/nifti.html
if (keyb == "mask"):
    img[i] = 255 * (img[i] - img[i].min()) / (img[i].max() - img[i].min())
img[i] = Image.fromarray(img[i]).convert("L").rotate(-90, expand = 1) # https://stackoverflow.com/questions/16720682/pil-cannot-write-mode-f-to-jpeg and https://www.geeksforgeeks.org/python-pil-paste-and-rotate-method/
img[i].save(files[ind[i]] + ".png") # https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image  
```
