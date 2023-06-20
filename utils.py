import os
from PIL import Image
from PIL import ImageFile
import requests
import base64
from io import BytesIO
import imghdr
import numpy as np
import onnxruntime
import torch
import cn_clip.clip as clip
from tqdm import tqdm
import json
import pandas as pd

clip_base = "中文CLIP(Base)"
clip_large = "中文CLIP(Large)"
clip_large_336 = "中文CLIP(Large,336分辨率)"
# description = "本项目为CLIP模型的中文版本DEMO，中文CLIP使用大规模中文数据进行训练（~2亿图文对），可用于图文检索和图像、文本的表征提取，应用于搜索、推荐等应用场景。注：检索rt<1秒，耗时主要源自一些审核机制。"
description = "本项目用于2023年自然语言处理的课设，基于Chinese-Clip搭建的文到图搜索Demo，可以用于图文检索。数据基于Flick8K-CN与自构建数据集，终端用的onnx模型，没有做相关优化，检索可能较慢。项目组成员:唐飞、田凯旭、陈宇轩、于博洋、胡祺、李润轩"
yes = "是"
no = "否"

# server_ip = os.environ.get("CLIP_SERVER_IP", "127.0.0.1")

# clip_service_url_d = {
#     clip_base: f'http://{server_ip}/knn-service',
#     clip_large: f'http://{server_ip}/knn-service-large',
#     clip_large_336: f'http://{server_ip}/knn-service-large-336'
# }

def decode_b64(image):

    decoded_data = base64.b64decode(image)
    # Determine the image format using the imghdr module
    img_format = imghdr.what(None, decoded_data)
    # Load the decoded image data into a PIL Image object, specifying the image format
    img = Image.open(BytesIO(decoded_data)).convert('RGB')
    # Convert the PIL Image object to a numpy array
    img_array = np.array(img)
    return img_array

def create_onnx_model(txt_onnx_model_path):
    txt_sess_options = onnxruntime.SessionOptions()
    txt_run_options = onnxruntime.RunOptions()
    txt_run_options.log_severity_level = 2
    txt_session = onnxruntime.InferenceSession(txt_onnx_model_path,
                                            sess_options=txt_sess_options,
                                            providers=["CUDAExecutionProvider"])
    return txt_session



def extract_text_feat(txt_session,text):
    text_features = []
    for i in range(len(text)):
        one_text = np.expand_dims(text[i].cpu().numpy(),axis=0)
        text_feature = txt_session.run(["unnorm_text_features"], {"text":one_text})[0] # 未归一化的文本特征
        text_feature = torch.tensor(text_feature)
        text_features.append(text_feature)
    # 拿到text_feature
    text_features = torch.squeeze(torch.stack(text_features),dim=1) # 4个特征向量stack到一起
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    # 转为torch.Tensor 并放在cuda
    text_feat_tensor = text_feature.cuda()
    return text_feat_tensor

def cal_text_image_simi(img_json):
    image_ids = []
    image_feats = []
    score_tuples = []
    with open(img_json, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            image_ids.append(obj['image_id'])
            image_feats.append(obj['feature'])
    # 成功获取image_feature特征
    image_feats_array = np.array(image_feats, dtype=np.float32)
    print("Finished loading image features.")
    return image_ids,image_feats,score_tuples,image_feats_array

def make_topk(img_tsv,img_json,txt_session,text,return_n):
    text_feat_tensor = extract_text_feat(txt_session,text)
    image_ids,image_feats,score_tuples,image_feats_array = cal_text_image_simi(img_json)
    idx = 0
    # 开始算相似度 文本与图片--->文搜图
    while idx < len(image_ids):
        img_feats_tensor = torch.from_numpy(image_feats_array[idx : min(idx + 10000, len(image_ids))]).cuda() # [batch_size, feature_dim]
        batch_scores = text_feat_tensor @ img_feats_tensor.t() # [1, batch_size]
        for image_id, score in zip(image_ids[idx : min(idx + 10000, len(image_ids))], batch_scores.squeeze(0).tolist()):
            score_tuples.append((image_id, score))
        idx += 10000
    top_k_predictions = sorted(score_tuples, key=lambda x:x[1], reverse=True)[:return_n]
    # 找到top_k_predictions 从train_imgs.tsv找到索引并返回
    imgs_dict = {t[0]: t[1] for t in top_k_predictions}
    # print(imgs_dict)
    df = pd.read_csv(img_tsv, sep='\t', header=None, names=['id', 'image_path'])
    df.set_index('id', inplace=True)
    result = {}
    for id, value in imgs_dict.items():
        image_path = df.loc[id]['image_path']
        if not pd.isna(image_path):
            result[id] = (Image.fromarray(decode_b64(image_path)), value)
    return result


