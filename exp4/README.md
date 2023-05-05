## Exp4: Backdoor Attacks on DNN

Please refer to [exp4 requirements](./%E4%B8%8A%E6%9C%BA%E5%AE%9E%E9%AA%8C%E5%9B%9B.docx) for more details.

Download work_dir and pth_files [here](https://drive.google.com/drive/folders/1zzJWZ2vlwAiaMR9RakUn-hKo3EM1qQ2Y?usp=share_link), which contains training logs, json files and .pth files.

Folder structure:
```
Computer-Vision-Experiment\exp4
│  exp4_backdoor_attacks.ipynb
│  
├─configs
│  │  vgg13_bn.py
│
├─datasets
├─modules
│  │  vgg.py
│  │  __init__.py
│
├─pth_files
├─tools
│  │  data_poison.py
│  │  test.py
│  │  train.py
│  │  __init__.py
│          
└─work_dir
```

As shown below, backdoor trigger is a 4x4 white square on the bottom right corner of the input image and attack target is airplane.

![](../images/exp4/trigger.png)

Results:

![](../images/exp4/print.png)
![](../images/exp4/curves.png)
