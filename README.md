# IAUnet
This repository contains the code for the paper:
<br>
[**IAUnet: Global Context-Aware Feature Learning for Person Re-Identification**](https://arxiv.org/pdf/2007.09357.pdf)
<br>
Ruibing Hou, Bingpeng Ma, Hong Chang,  Xinqian Gu, Shiguang Shan, Xilin Chen
<br>
TNNLS 2020


### Abstract

Person re-identification (reID) by CNNs based networks has achieved favorable performance in recent years. However, most of existing CNNs based methods do not take full advantage of spatial-temporal context modeling. In fact, the global spatial-temporal context can greatly clarify local distractions to enhance the target feature representation. To comprehensively leverage the spatial-temporal context information, in this work, we present a novel block, Interaction-AggregationUpdate (IAU), for high-performance person reID. Firstly, SpatialTemporal IAU (STIAU) module is introduced. STIAU jointly incorporates two types of contextual interactions into a CNN framework for target feature learning. Here the spatial interactions learn to compute the contextual dependencies between different body parts of a single frame. While the temporal interactions are used to capture the contextual dependencies between the same body parts across all frames. Furthermore, a Channel IAU (CIAU) module is designed to model the semantic contextual interactions between channel features to enhance the feature representation, especially for small-scale visual cues and body parts. Therefore, the IAU block enables the feature to incorporate the globally spatial, temporal, and channel context. It is lightweight, end-to-end trainable, and can be easily plugged into existing CNNs to form IAUnet. The experiments show that IAUnet performs favorably against state-of-the-art on both image and video reID tasks and achieves compelling results on a general object categorization task.

### Training and test

  ```Shell
  # For Market
  1. we first generate the part masks with the code https://github.com/Engineering-Course/LIP_JPPNet/.
  
  ```

### Citation

If you use this code for your research, please cite our paper:
```
@article{IAUnet,
  title={IAUnet: Global Context-Aware Feature Learning for Person Re-Identification},
  author={Ruibing Hou and Bingpeng Ma and Hong Chang and Xinqian Gu and Shiguang Shan and Xilin Chen},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2020},
  publisher={IEEE}
}
```

### Platform
This code was developed and tested with pytorch version 1.0.1.


## Acknowledgments

This code is based on the implementations of [**Deep person reID**](https://github.com/KaiyangZhou/deep-person-reid/tree/master/torchreid).
