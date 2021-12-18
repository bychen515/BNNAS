BN-NAS: Neural Architecture Search with Batch Normalization
=========================================
Code for [BN-NAS: Neural Architecture Search with Batch Normalization](https://arxiv.org/abs/2108.07375) accepted by ICCV2021

This project is the re-implementation based on [ABS](https://github.com/megvii-model/AngleNAS) and [SPOS](https://github.com/megvii-model/SinglePathOneShot).

## Requirements
- Pytorch 1.3
- Python 3.5+
- [Apex](https://github.com/NVIDIA/apex)

The requirements.txt file lists other Python libraries that this project depends on, and they will be installed using:
pip3 install -r requirements.txt

## Thanks
This implementation of BNNAS is based on [ABS](https://github.com/megvii-model/AngleNAS) and [SPOS](https://github.com/megvii-model/SinglePathOneShot). Please ref to their reposity for more details.

## Citation
If you find that this project helps your research, please consider citing our paper:
@inproceedings{chen2021bn,
  title={Bn-nas: Neural architecture search with batch normalization},
  author={Chen, Boyu and Li, Peixia and Li, Baopu and Lin, Chen and Li, Chuming and Sun, Ming and Yan, Junjie and Ouyang, Wanli},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={307--316},
  year={2021}
}
