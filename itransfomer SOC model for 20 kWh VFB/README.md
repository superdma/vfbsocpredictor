# iTransfomer SOC model for 20 kWh VFB system


This is the source code for iTransfomer-based SOC model for 20 kWh VFB system. The open-source library of  [iTransformer](https://arxiv.org/abs/2310.06625) was applied.

## Dataset for SOC

The dataset for SOC was included in ./dataset/SOC/Dataset_charge.csv and ./dataset/SOC/Dataset_discharge.csv.

## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data.
2. Train and evaluate model. We provide the experiment scripts for SOC prediction. You can reproduce the experiment results as the following examples:

```
# long-term forecast
bash ./sub.sh
```

## Citation

If you find this repo useful, please cite this paper and the [iTransformer](https://arxiv.org/abs/2310.06625) paper

## Contact
If you have any questions or suggestions, feel free to contact:

- Tianyu Li (litianyu@dicp.ac.cn)
- Jiawei Sun (sunjiawei@dicp.ac.cn)
- Shengkai Xu (xushengkai@vip.qq.com)
- Feng Xing (xingf@dicp.ac.cn)
- Zhao Ma (zhao_ma@qq.com)
- He Jiang (jianghe@dlut.edu.cn)
- Zonghao Liu (zonghao.liu@rongkepower.com)
- Tao Liu (liutao@dicp.ac.cn)
- Xianfeng Li* (lixianfeng@dicp.ac.cn)

## Acknowledgement

This work was supported by the Key R&D projects of National Natural Science Foundation of China (2022YFB2404904), National Natural Science Foundation of China (22309178), Strategic Priority Research Program of the CAS (XDA0400402), DICP funding (DICP I202239), Liaoning International Cooperation Project (2023JH2/10700002). The AI-driven experiments, simulations and model training were performed on the robotic AI-Scientist platform of Chinese Academy of Sciences.
