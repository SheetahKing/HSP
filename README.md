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
*Main results1:*

| Seeds | Model | Ref. | FB15K-DB15K | | | FB15K-YG15K | | |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| | | | Hits@1 | Hits@10 | MRR | Hits@1 | Hits@10 | MRR |
| **20%** | PoE | ESWC'19 | 0.126 | 0.251 | 0.170 | 0.250 | 0.495 | 0.334 |
| | MMEA | KSEM'20 | 0.265 | 0.541 | 0.357 | 0.234 | 0.480 | 0.317 |
| | EVA | AAAI'21 | 0.199 | 0.448 | 0.283 | 0.153 | 0.361 | 0.224 |
| | MCLEA | COLING'22 | 0.295 | 0.582 | 0.393 | 0.254 | 0.484 | 0.332 |
| | MEAformer | MM'23 | 0.417 | 0.715 | 0.518 | 0.327 | 0.595 | 0.417 |
| | PCMEA† | AAAI'24 | 0.436 | 0.707 | 0.528 | 0.397 | 0.654 | 0.486 |
| | LoginMEA† | ECAI'24 | 0.536 | 0.784 | 0.667 | 0.572 | 0.758 | 0.652 |
| | OTMEA | ICASSP'25 | 0.443 | 0.712 | 0.523 | 0.395 | 0.628 | 0.477 |
| | GSIEA | IPM'25 | **0.580** | 0.791 | 0.649 | 0.515 | 0.728 | 0.588 |
| | RICEA | ACL'25 | 0.471 | 0.720 | 0.557 | 0.411 | 0.658 | 0.497 |
| | CDMEA† | SIGIR'25 | 0.549 | 0.741 | 0.611 | 0.434 | 0.682 | 0.523 |
| | ME³A-N | IPM'25 | 0.492 | 0.824 | 0.689 | 0.587 | 0.766 | 0.651 |
| | ME³A-M | IPM'25 | 0.502 | 0.822 | 0.683 | 0.586 | 0.764 | 0.659 |
| | **HSP-N** | - | 0.544 | 0.831 | 0.699 | 0.597 | 0.784 | **0.679** |
| | **HSP-M** | - | 0.542 | **0.837** | **0.702** | **0.609** | **0.791** | 0.672 |
| | Δ Avg. | - | -0.036 | 0.013 | 0.013 | 0.022 | 0.025 | 0.020 |
| | Δ Std. | - | ±.012 | ±.009 | ±.010 | ±.011 | ±.007 | ±.009 |
| **50%** | PoE | ESWC'19 | 0.464 | 0.658 | 0.533 | 0.411 | 0.669 | 0.498 |
| | MMEA | KSEM'20 | 0.417 | 0.703 | 0.512 | 0.403 | 0.645 | 0.486 |
| | EVA | AAAI'21 | 0.334 | 0.589 | 0.422 | 0.311 | 0.534 | 0.388 |
| | MCLEA | COLING'22 | 0.555 | 0.784 | 0.637 | 0.501 | 0.705 | 0.574 |
| | MEAformer | MM'23 | 0.619 | 0.843 | 0.698 | 0.560 | 0.778 | 0.639 |
| | PCMEA† | AAAI'24 | 0.633 | 0.825 | 0.701 | 0.606 | 0.785 | 0.652 |
| | LoginMEA† | ECAI'24 | 0.712 | 0.857 | 0.755 | 0.648 | 0.805 | 0.699 |
| | OTMEA | ICASSP'25 | 0.658 | 0.854 | 0.734 | 0.617 | 0.809 | 0.687 |
| | GSIEA | IPM'25 | 0.716 | 0.886 | 0.775 | 0.646 | 0.828 | 0.711 |
| | RICEA | ACL'25 | 0.648 | 0.852 | 0.721 | 0.617 | 0.811 | 0.687 |
| | CDMEA† | SIGIR'25 | 0.675 | 0.861 | 0.737 | 0.628 | 0.824 | 0.692 |
| | ME³A-N | IPM'25 | 0.729 | 0.884 | 0.788 | 0.664 | 0.827 | 0.721 |
| | ME³A-M | IPM'25 | 0.735 | 0.882 | 0.797 | 0.662 | 0.826 | 0.726 |
| | **HSP-N** | - | 0.747 | 0.893 | 0.804 | 0.674 | 0.837 | 0.736 |
| | **HSP-M** | - | **0.755** | **0.894** | **0.807** | **0.681** | **0.841** | **0.739** |
| | Δ Avg. | - | 0.020 | 0.008 | 0.010 | 0.017 | 0.013 | 0.013 |
| | Δ Std. | - | ±.007 | ±.005 | ±.007 | ±.008 | ±.007 | ±.006 |
| **80%** | PoE | ESWC'19 | 0.666 | 0.820 | 0.721 | 0.492 | 0.705 | 0.572 |
| | MMEA | KSEM'20 | 0.590 | 0.869 | 0.685 | 0.598 | 0.839 | 0.682 |
| | EVA | AAAI'21 | 0.484 | 0.696 | 0.563 | 0.491 | 0.692 | 0.565 |
| | MCLEA | COLING'22 | 0.735 | 0.890 | 0.790 | 0.667 | 0.824 | 0.722 |
| | MEAformer | MM'23 | 0.765 | 0.916 | 0.820 | 0.703 | 0.873 | 0.766 |
| | PCMEA† | AAAI'24 | 0.744 | 0.904 | 0.818 | 0.715 | 0.869 | 0.784 |
| | LoginMEA† | ECAI'24 | 0.773 | 0.908 | 0.822 | 0.724 | 0.904 | 0.803 |
| | OTMEA | ICASSP'25 | 0.793 | 0.919 | 0.839 | 0.739 | 0.898 | 0.798 |
| | GSIEA | IPM'25 | 0.804 | 0.928 | 0.847 | 0.734 | 0.861 | 0.793 |
| | RICEA | ACL'25 | 0.776 | 0.916 | 0.829 | 0.734 | 0.892 | 0.792 |
| | CDMEA† | SIGIR'25 | 0.804 | 0.931 | 0.859 | 0.743 | 0.891 | 0.806 |
| | ME³A-N | IPM'25 | 0.824 | 0.933 | 0.841 | 0.748 | 0.935 | 0.808 |
| | ME³A-M | IPM'25 | 0.822 | 0.928 | 0.844 | 0.751 | 0.931 | 0.811 |
| | **HSP-N** | - | 0.835 | 0.935 | 0.858 | **0.767** | 0.943 | **0.821** |
| | **HSP-M** | - | **0.838** | **0.937** | **0.861** | 0.766 | **0.946** | 0.819 |
| | Δ Imp. | - | 0.014 | 0.004 | 0.002 | 0.016 | 0.011 | 0.010 |
| | Δ Std. | - | ±.006 | ±.005 | ±.004 | ±.006 | ±.004 | ±.005 |

*Main results2:*

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
