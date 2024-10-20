# UnSeg: One Universal Unlearnable Example Generator is Enough against All Image Segmentation [NeurIPS 2024]

## Abstract

Image segmentation is a crucial vision task that groups pixels within an image into semantically meaningful segments, which is pivotal in obtaining a fine-grained understanding of real-world scenes. However, an increasing privacy concern exists regarding training large-scale image segmentation models on unauthorized private data. In this work, we exploit the concept of unlearnable examples to make images unusable to model training by generating and adding unlearnable noise into the original images. Particularly, we propose a novel Unlearnable Segmentation (UnSeg) framework to train a universal unlearnable noise generator that is capable of transforming any downstream images into their unlearnable version. The unlearnable noise generator is finetuned from the Segment Anything Model (SAM) via bilevel optimization on an interactive segmentation dataset towards minimizing the training error of a surrogate model that shares the same architecture with SAM but is trained from scratch. We empirically verify the effectiveness of UnSeg across 6 mainstream image segmentation tasks, 10 widely used datasets, and 7 different network architectures, and show that the unlearnable images can reduce the segmentation performance by a large margin. Our work provides useful insights into how to leverage foundation models in a data-efficient and computationally affordable manner to protect images against image segmentation models. 
<p align="center">
  <img src="UnSeg.png" width="700"/>
</p>

## :rocket: Updates
* **[TODO]** Code is coming soon. Thanks for your attention and patience.

* **[2024/10/13]** Our paper is available in [[Arxiv](https://arxiv.org/abs/2410.09909)].

## Citation
If you find UnSeg useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{sun2024unseg,
  title={UnSeg: One Universal Unlearnable Example Generator is Enough against All Image Segmentation},
  author={Sun, Ye and Zhang, Hao and Zhang, Tiehua and Ma, Xingjun and Jiang, Yu-Gang},
  booktitle={NeurIPS},
  year={2024}
}
```
