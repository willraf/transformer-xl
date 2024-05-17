# Setup 

## 1. Change data.sh to prevent error
There was an issure with getdata.sh where datasets 'wikitext-2' and 'WikiText-103 (WT2)' fail to load. This causes subsequent datasets to be loaded to the wrong directory. 
Fixed by simply commenting out the code to load these datasets, as we will not be using them. 

## 2. Environment
The reproducability scripts suggest 'Make sure the machine have 4 GPUs, each with at least 11G memory'
I therefore ran these scripts on the ecs GPU-server. 
Details on the environment requirements are vague at best. There is no requirements.txt file provided, and the only prerequesits stated on github readme (https://github.com/kimiyoung/transformer-xl) is 

`Pytorch 0.4: conda install pytorch torchvision -c pytorch`

Here is the code I used to access the server and set up the environment:

`ssh wr1g20@yann.ecs.soton.ac.uk`

Install miniconda

`conda create --name transformer-xl python==3.7`

`conda activate transformer-xl`

`conda install pytorch==0.4.1`

`conda install pytorch==0.4.1 torchvision==0.2.1 cuda90 -c pytorch`

`conda install scipy`

Now running the demo scripts should work:

`git clone https://github.com/willraf/transformer-xl.git`

`cd transformer-xl`

`bash getdata.sh` **FIXED

`cd pytorch`

`bash run_enwik8_base.sh train --work_dir ~/xl-exp1`

In order to reprodce the results given by the paper, we need to make some changes.

# Run

## Text8
Baseline:

`bash run_text8_base.sh train --work_dir ~/xl-exp1`

`bash run_text8_base.sh eval --work_dir ~/xl-exp1`


RSA Version:
`bash run_text8_rsa.sh train --work_dir ~/xl-exp1`

`bash run_text8_rsa.sh eval --work_dir ~/xl-exp1`

## Wiki8

Same as above, but with run_enwik8_base.sh and run_enwik8_rsa.sh

# Code
The majority of code modified was within rem.py, mem_transformer.py and train.py. Added sections are often preceded with a '#!' comment. 
