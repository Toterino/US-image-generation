This repository represents the code used for the following paper: [...] 

https://github.com/Toterino/stable-diffusion

https://github.com/Toterino/ControlNet

# Stable Diffusion
The easiest way to finetune stable diffusion is to have a look at Justin Pinkney's [repository](https://github.com/justinpinkney/stable-diffusion), a ML researcher, and their corresponding [guide](https://github.com/LambdaLabsML/examples/tree/main/stable-diffusion-finetuning). Since I was having trouble finetuning on the original repository and I wanted to understand what made Pinkey's code work, I slowly added a few pieces of their code to the official repository, slightly tweaked them and it worked.

The following guide will thus only be for reproducibility purposes. Please follow Pinkney's guide!


TODO: mention the changes

# ControlNet
The official repository for ControlNet has a [guide](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md) on how to train on your own data. We will thus only explain our setup for reproducibility purposes. Please follow the official guide!

### _root_dir_
- We created the following folder structure and inserted the images in the corresponding folders:
```
training/data/source/insert_conditions_here.png
training/data/target/insert_dataset_here.png
```
- We created a prompts.json file in ```training/data``` with the following format (the project json files are provided):
  
```
{"source": "source/normal (1)_mask.png", "target": "target/normal (1).png", "prompt": "Ultrasound image of normal breast"}
{"source": "source/normal (2)_mask.png", "target": "target/normal (2).png", "prompt": "Ultrasound image of normal breast"}
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### _tutorial_dataset.py_
- We adjusted the following lines to include the correct path:
  
```python 
with open('./training/BUSI/prompt_busi.json', 'rt') as f:
```

```python 
source = cv2.imread('./training/BUSI/' + source_filename) # "source" is the condition
```

```python 
target = cv2.imread('./training/BUSI/' + target_filename) # "target" is the data
```

- We resized the dataset images to the same shape (512x512 in our case):
```python 
source = cv2.resize(source, (512,512), interpolation= cv2.INTER_LINEAR) # https://learnopencv.com/image-resizing-with-opencv/#resize-by-wdith-height
target = cv2.resize(target, (512,512), interpolation= cv2.INTER_LINEAR) # https://learnopencv.com/image-resizing-with-opencv/#resize-by-wdith-height
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### _root_dir_
- We downloaded stable diffusion 1.5 from the hugging face [page](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) (v1-5-pruned.ckpt) and placed the model in ```./models/```
- We ran the following terminal command, which initializes a CN model
```python
python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/old_busi_control_sd15_ini.ckpt
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### _tutorial_train.py_
- We adjusted the hyperparameters and code:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Use only GPU #3

resume_path = './models/old_camus_control_sd15_ini.ckpt' # Path to the init model
batch_size = 4 # Batch size for training
logger_freq = 5000 # After how many steps should the model sample
learning_rate = 1e-5 # Learning rate used for tarining
sd_locked = True # Keep the weights of SD frozen
only_mid_control = False
```
- We made it so that all checkpoints are saved after a certain number of steps:
```python
# Making a checkpoint callback that saves every model after a certain number of steps
call_checkpoint = ModelCheckpoint( # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
    every_n_train_steps = 5000,  # Every 5000 steps
    save_top_k=-1  # All checkpoints are saved
)

dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, call_checkpoint]) 
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### _root_dir_
- We ran the following terminal command, which trains the model:
```python
CUDA_VISIBLE_DEVICES=3 python tutorial_train.py
```
