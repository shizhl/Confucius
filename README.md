# Confucius-Tool-Learning

<div align=center>
	<img src="./README.assets/image-20230817114433957.png"/>
</div>

Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum.

## News

- [2023.12.9] Our work are accepted via AAAI 2024!
- [2023.8.27] Our paper is now available at https://arxiv.org/abs/2308.14034

## Dataset

We collect a tool-use dataset via Self-Instruct, i.e., prompting ChatGPT to generate tool-use sample automatically.


### Data description
```json
{   
  "api": "The api for solve the specific task",
  "number": "The number for calling API in this case",
  "prompt": "The prompt for generating this example",
  "task": "The task name",
  "question": "The specific query based on the API in the this task",
  "_answer": "The solution to solve problem in the format of chain of thought (COT), where the above APIs are called back. (Optional)"
}
```
A concrete example:

```json
{
    "api": [
        [
            "CAL",
            "expression: 2500/5",
            "CAL(expression: e)->float: calculate the result of expression `e`, e.g. 1+2, 1/3, 4*5 and 7-1."
        ],
        [
            "CAL",
            "expression: 2*%s1",
            "CAL(expression: e)->float: calculate the result of expression `e`, e.g. 1+2, 1/3, 4*5 and 7-1."
        ],
        [
            "CAL",
            "expression: %s2-200",
            "CAL(expression: e)->float: calculate the result of expression `e`, e.g. 1+2, 1/3, 4*5 and 7-1."
        ]
    ],
    "number": 3,
    "prompt": "According to the ratio, for every 5 parts that Johnson gets, Mike gets 2 parts.Since Johnson got $2500, each part is therefore $2500/5 = $<<2500/5=500>>500.Mike will get 2*$500 = $<<2*500=1000>>1000.After buying the shirt he will have $1000-$200 = $<<1000-200=800>>800 left. ### 800",
    "question": "The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5 respectively. If Johnson got $2500, how much will Mike have after spending some of his share on a shirt that costs $200?",
    "_answer": "According to the ratio, for every 5 parts that Johnson gets, Mike gets 2 parts. Since Johnson got $2500, each part is therefore [CAL(2500/5) -> %s1].Mike will get 2*$%s1 = [CAL(2*%s1) -> %s2]. After buying the shirt, he will have $%s2-$200 = [CAL(%s2-200) -> %s3] left. ### 800",
    "task": "calculation"
}
```

## Train

```txt
torchrun --nnodes <num of machine> --nproc_per_node <number of device per machine>  --master_port <port>  \
train.py --per_device_train_batch_size <batch size> \
      --num_device_per_node <number of device>  \
      --num_works   \
      --strategy <strategy for speeding > \
      --gradient_accumulation_steps <16 for default> \
      --model_name_or_path <huggingface model path > \
      --train_data_path <data path for training (json/jsonl format)> \
      --warm_up <int, the number of training sample in warm up stage> \
      --in_domain <int, the number of training sample in in-category stage> \
      --cross_domain <int, the number of training sample in cross-category stage> 
```



```txt
torchrun --nnodes 1 --nproc_per_node  4  --master_port 9994 \
 main.py --per_device_train_batch_size 4 \
      --num_device_per_node 4 \
      --num_works 10 \
      --gradient_accumulation_steps 32 \
      --model_name_or_path llama \
      --train_data_path   ../train.v4.151074.json  \
      --output_dir  \
      --warm_up 0 --in_domain 5000 --cross_domain 5000 \
      --max_epochs 25 \
```

the successful runing state is:

![image-20230817165314476](README.assets/image-20230817165314476.png)



## Inference

We provide the following command to conduct the inference.

```txt
python inference.py \
--model_name_or_path llama \
--output_file <path to store the model output> \
--huggingface_ckpt_path <the path used to restore the huggingface-style weight>  \
--n <the number of examples used for in-context learning> \
--data_path <the test dataset> \
--ranks  <the device id of gpus, which can inference with multiple GPU devices>
```

the successful runing state is:

![image-20230818085121242](README.assets/image-20230818085121242.png)


## Environment Set up

1. python 3.9 
2. pytorch lightning (1.9.0)
3. Deepspeed (deepspeed in pytorch lightning)
4. transformer (install from source)
5. pytorch (torch 1.11)
6. tqdm
7. openai (only for collecting data)

We optimize the model using deepspeed ZeRO-three  strategy with the learning rate of $5e^{-5}$ and the weight decay coefficient of 0.01.
We use **4 NVIDIA A100-PCIE-80GB GPUs** to train our model.

## Todo

- [ ] The code and dataset will be released as soon as possible.

For any questions or requests, feel free to contact me at shizhl@mail.sdu.edu.cn. 

## Other work

We also release the `Fuzi-Mingcha`, a Chinese legal LLM, which has shown strong performance in legal tasks. Fuzi-Mingcha is jointly developed by Shandong University, Inspur, and China University of Political Science and Law, which is trained based on massive Chinese unsupervised judicial corpus and supervised judicial fine-tuning data using ChatGLM as backbone. It supports law search, case analysis, trinitarian reasoning judgment and judicial dialog, aiming to provide users with all-round and highly accurate legal consultation and answer services.

Click [here](https://github.com/irlab-sdu/fuzi.mingcha) for more details.  Thanks a lot for all the prior works.

## Citation

```
@inproceedings{Gao2023ConfuciusIT,
title={Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum},
author={Shen Gao and Zhengliang Shi and Minghang Zhu and Bowen Fang and Xin Xin and Pengjie Ren and Zhumin Chen and Jun Ma and Zhaochun Ren},
booktitle={AAAI},
year={2024}
}
```
