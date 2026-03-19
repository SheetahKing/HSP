# HSP
We sincerely appreciate for the previous paper "[ME3A: A Multimodal Entity Entailment framework for multimodalEntity Alignment](https://github.com/OreOZhao/ME3A)", our work is based on its opened source. 

the code for HSP paper


## Model Architecture Diagram 模型架构图
![the structure of the HSP model](./model_01.png)

## Baseline Model基线模型
- **PoE** establishes a foundational framework for cross-modal semantic fusion;
- **MMEA** unifies structure, attributes, and visual information for multimodal entity alignment;
- **EVA** employs vision as a pivot for unsupervised alignment;
- **MCLEA** advances multimodal contrastive learning to align heterogeneous embedding spaces;
- **MEAformer** formulates a meta-modality hybrid mechanism for heterogeneous modality fusion;
- **PCMEA** introduces pseudo-label calibration to alleviate noise propagation;
- **LoginMEA** presents a local-to-global interaction network for multi-level relational fusion;
- **OTMEA** aligns multimodal distributions via optimal transport to reduce heterogeneity;
- **GSIEA** separately models graph structure and multimodal information to mitigate structural heterogeneity;
- **RICEA** enhances generalization through relative interaction modeling and dynamic calibration;
- **CDMEA** addresses modality bias from a causal perspective;
- **ME³A** formulates MEA as an entity entailment task to enable fine-grained cross-graph interaction.

## Experimental Results实验结果
| Method | Ref. | DBP15K_ZH-EN | | | DBP15K_JA-EN | | | DBP15K_FR-EN | | |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | Hits@1 | Hits@10 | MRR | Hits@1 | Hits@10 | MRR | Hits@1 | Hits@10 | MRR |
| EVA | AAAI'21 | 0.680 | 0.910 | 0.762 | 0.673 | 0.908 | 0.757 | 0.683 | 0.923 | 0.767 |
| MCLEA | COLING'22 | 0.715 | 0.923 | 0.788 | 0.715 | 0.909 | 0.785 | 0.711 | 0.909 | 0.782 |
| MSNEA | WWW'23 | 0.601 | 0.83 | 0.684 | 0.535 | 0.775 | 0.617 | 0.543 | 0.801 | 0.630 |
| MEAformer | MM'23 | 0.771 | 0.951 | 0.835 | 0.764 | 0.959 | 0.834 | 0.770 | 0.961 | 0.841 |
| PSNEA | MM'23 | 0.816 | 0.957 | 0.869 | 0.819 | 0.963 | 0.853 | 0.844 | 0.982 | 0.891 |
| DESAlign | ICDE'24 | 0.868 | 0.969 | 0.909 | 0.871 | 0.974 | 0.913 | 0.882 | 0.982 | 0.924 |
| IBMEA | MM'24 | 0.859 | 0.975 | 0.903 | 0.856 | 0.978 | 0.902 | 0.864 | 0.985 | 0.911 |
| LoginMEA † | ECAI'24 | 0.877 | 0.972 | 0.905 | 0.868 | 0.972 | 0.910 | 0.879 | 0.983 | 0.919 |
| OTMEA | ICASSP'25 | 0.779 | 0.953 | 0.842 | 0.783 | 0.959 | 0.849 | 0.796 | 0.97 | 0.861 |
| AMFSEA † | COLING'25 | 0.691 | 0.879 | 0.751 | 0.696 | 0.871 | 0.757 | 0.767 | 0.914 | 0.818 |
| ME3A-N | IPM'25 | 0.894 | 0.981 | 0.916 | 0.878 | 0.962 | 0.927 | 0.893 | 0.994 | 0.935 |
| ME3A-M | IPM'25 | 0.886 | 0.984 | 0.919 | 0.882 | 0.970 | 0.929 | 0.895 | 0.994 | 0.938 |
| **HSP-N** | - | 0.897 | **0.987** | 0.924 | **0.894** | 0.982 | 0.936 | 0.903 | 0.993 | 0.944 |
| **HSP-M** | - | **0.902** | 0.986 | **0.930** | 0.889 | **0.985** | **0.938** | **0.906** | **0.995** | **0.946** |
| Δ Imp. | - | 0.005 | 0.001 | 0.006 | 0.005 | 0.003 | 0.002 | 0.003 | 0.001 | 0.002 |
| Δ Std. | - | ±.004 | ±.003 | ±.005 | ±.004 | ±.003 | ±.002 | ±.003 | ±.002 | ±.004 |

## Dataset数据集
*Cross-KG datasets:* The original cross-KG datasets (FB15K-DB15K/YAGO15K) comes from: [https://github.com/mniepert/mmkb]
*Bilingual datasets:* The multi-modal version of DBP15K dataset comes from the: [https://github.com/cambridgeltl/eva]
