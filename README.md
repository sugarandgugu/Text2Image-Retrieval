# 基于Clip、Chinese-Clip搭建的文搜图Demo
![这是图片](./image/title.png"Magic Gardens")
## 数据处理(构建自己的数据集)

由于Chinese-clip库并没有教程构建自己的数据集，这里为了用Chinese-clip库进行微调，将自己构建的数据进行了对齐，其中

Flickr8K-CN数据集已经给出(数据来源<https://github.com/bubbliiiing/clip-pytorch>)。Chinese-clip的图片与文本都有一个id，但是下列给出的数据集并没有id。要用Chinese-clip训练自己构建的数据集，必须满足其库的数据集要求。其中Flickr8K-CN的json文件格式如下，image代表图片的路径，caption代表图片的描述。构建自己的数据库，请使用该库下面的process.ipynb。运行后会生成包含id的图片(base\_64格式)、生成包含id的文本。

```markdown
[
  {
    "image": "flickr8k-images/2513260012_03d33305cf.jpg",
    "caption": [
      "A black dog is running after a white dog in the snow .",
      "Black dog chasing brown dog through snow",
      "Two dogs chase each other across the snowy ground .",
      "Two dogs play together in the snow .",
      "Two dogs running through a low lying body of water ."
    ]
  },
]
```

其中利用process.ipynb生成后的文本、图片如下所示。

```markdown
# 文本形式
{"text_id": 1, "text": "在玩滑板的两个女孩。", "image_ids": [1]}
# 图片tsv格式
id  					image
1   					img的base64格式
```

经过上述过程，我们已经包含包含图片与文本的tsv、jsonl文件，如下所示。由于训练的时候Chinese-clip这个库需要利用到lmdb数据库，我们需要把下列文件转成其对应的形式。利用如下脚本。

```markdown
├── train_imgs.tsv      # 图片id & 图片内容
├── train_texts.jsonl   # 文本id & 文本内容，连同匹配的图片id列表
├── valid_imgs.tsv
├── valid_texts.jsonl

# DATAPATH代表你创建的文件夹名字,假如你的文件夹叫Flickr8K-CN。其结构可以如下列所示。
Flickr8K-CN
	datasets
		Flickr8K-CN
			├── train_imgs.tsv      # 图片id & 图片内容
			├── train_texts.jsonl   # 文本id & 文本内容，连同匹配的图片id列表
			├── valid_imgs.tsv
			├── valid_texts.jsonl	
			lmdb
				train
					imgs
					pair
				valid	
					imgs
					pair	
```

```python
# 注意你先需要把Chinese-clip拉取下来，本次Demo是基于Chinese-clip构建的。
python cn_clip/preprocess/build_lmdb_dataset.py \
    --data_dir ${DATAPATH}/datasets/${dataset_name}
    --splits train,valid

# 把DATAPATH换成自己的
python cn_clip/preprocess/build_lmdb_dataset.py \
    --data_dir Flickr8K-CN/datasets/Flickr8K-CN
    --splits train,valid
```

至此，我们已经把自己的数据集构建成[Chinese-clip]()对应的数据格式，下面可以进行模型微调。

## 模型微调

假设您已经拉取好了[Chinese-clip](https://github.com/OFA-Sys/Chinese-CLIP)这个库,根据上述流程已经构建好了自己的数据集。下面需要下载预训练权重，放在与datasets文件夹同一目录的pretrained\_weights文件夹下(自己创建)。其中[CN-CLIP]()[~ViT-B/16~](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt)~给出，如需要其他权重需要去CN-clip的官方库下载。~在Chinese-clip根目录下，运行如下代码即可微调。

```markdown
cd Chinese-CLIP/ 
# 把DATAPATH换成自己的
bash run_scripts/flickr30k_finetune_vit-b-16_rbt-base.sh ${DATAPATH}
bash run_scripts/flickr30k_finetune_vit-b-16_rbt-base_flip.sh ${DATAPATH}
```

注意: 如果您的机器是单卡，scipts里面的内容需要设置为单卡而不是分布式，参考以下配置。

```markdown
GPUS_PER_NODE=1

WORKER_CNT=1

export MASTER_ADDR=localhost

export MASTER_PORT=8514

export RANK=0 
```

微调了之后，在我们存放数据集的文件夹出现experiments的文件，结构如下所示。

```markdown
Flickr8K-CN
	datasets
		Flickr8K-CN
	experiments
		flickr30k_finetune_vit-b-16_roberta-base_bs128_8gpu	
			checkpoints
				epoch1.pt
```

## 部署思路

在你转为ONNX文件后，只需要在我们的代码放入以下有个文件即可部署。

```python
img_json = '/root/autodl-tmp/Chinese-CLIP/cn_clip/Text2Image_deploy/data/train_imgs.img_feat.jsonl'
img_tsv = '/root/autodl-tmp/Chinese-CLIP/cn_clip/Text2Image_deploy/data/train_imgs.tsv'
```











