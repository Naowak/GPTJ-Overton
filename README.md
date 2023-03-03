# GPTJ-Overton
Codes and documentation needed to train GPTJ (or fr-boris).

### Steps to train model

Create an instance composed of 2 NVIDIA A100 40GB VRAM and 170 GB of RAM on Google Cloud Platform.  

```
gcloud compute instances create gpuserver \
   --project overton-377516 \
   --zone asia-northeast3-b \
   --machine-type=a2-highgpu-2g \
   --maintenance-policy TERMINATE \
   --image-family pytorch-1-7-cu110 \
   --image-project deeplearning-platform-release \
   --boot-disk-size 200GB \
   --accelerator="type=nvidia-tesla-a100,count=2" \
   --metadata="install-nvidia-driver=True" \
   --preemptible
```

If you pre-configured you account on GCP, you can connect with ssh to the VM  
```
ssh naowak@IP-ADRRESS 
```

Clone the repo and its submodules
```
git clone --recursive git@github.com:Naowak/GPTJ-Overton.git
```

Install the requirements, and this version of transformers
```
pip install -r requirements.txt
pip install git+https://github.com/StellaAthena/transformers
```

Give all rights to all files in finetune-gpt and go inside
```
chmod -R 777 finetune-gpt/
cd finetune-gpt
```

Transfer training data to remote machine (run command from local machine)
```
scp train.txt validation.txt IP-ADDRESS:/home/naowak/finetune-gpt/
```

Tranform train.txt and validate.txt in train.csv and validate.csv
```
python text2csv.py
```

Run the training on Cedille/fr-boris model
```
deepspeed --num_gpus=2 run_clm.py \
--deepspeed ds_config.json \
--model_name_or_path Cedille/fr-boris \
--train_file train.csv \
--validation_file validation.csv \
--do_train \
--do_eval \
--fp16 \
--overwrite_cache \
--evaluation_strategy="steps" \
--output_dir finetuned \
--eval_steps 200 \
--num_train_epochs 1 \
--gradient_accumulation_steps 2 \
--per_device_train_batch_size 16
```

Monitor the training, enter those commands in two others terminal to see CPU & GPU usage
```
watch -n0.5 nvidia-smi
htop
```

Retrieve files from remote
```
scp -r IP-ADDRESS:/home/naowak/finetune-gpt/finetuned/ ./model_trained
```

### Steps to infer

Create the VM in command line with GCP
```
gcloud compute instances create gpuserver \
   --project overton-377516 \
   --zone europe-west1-b \
   --machine-type=custom-6-49152-ext \
   --maintenance-policy TERMINATE \
   --image-family pytorch-1-7-cu110 \
   --image-project deeplearning-platform-release \
   --boot-disk-size 100GB \
   --accelerator="type=nvidia-tesla-t4,count=1" \
   --metadata="install-nvidia-driver=True" \
   --preemptible
```

Copy the model from local to remote VM
```
scp -r ./finetune-gpt/finetuned-model naowak@IP-ADDRESS:/home/naowak/
```

Clone the repo and its submodules
```
git clone --recursive git@github.com:Naowak/GPTJ-Overton.git
```

Install the requirements, and this version of transformers
```
cd GPTJ-Overton
pip install -r requirements.txt
cd transformers
pip install .
cd ..
```

Give all rights to all files in finetune-gpt, move model inside and go inside
```
chmod -R 777 finetune-gpt/
mv finetuned-model finetune-gpt/
cd finetune-gpt
```

Make predictions, generate text :
```
python run_generate.py finetuned_model/
```