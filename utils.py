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

clip_base = "中文CLIP(Base)"
clip_large = "中文CLIP(Large)"
clip_large_336 = "中文CLIP(Large,336分辨率)"
description = "本项目为CLIP模型的中文版本DEMO，中文CLIP使用大规模中文数据进行训练（~2亿图文对），可用于图文检索和图像、文本的表征提取，应用于搜索、推荐等应用场景。注：检索rt<1秒，耗时主要源自一些审核机制。"

yes = "是"
no = "否"

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
                                           )
    return txt_session
txt_session = create_onnx_model('checkpoint/vit-b-16.txt.fp16.onnx')
# 为4条输入文本进行分词。序列长度指定为52，需要和转换ONNX模型时保持一致（参见转换时的context-length参数）
text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], context_length=52) 

# 用ONNX模型依次计算文本侧特征
text_features = []
for i in range(len(text)):
    one_text = np.expand_dims(text[i].cpu().numpy(),axis=0)
    text_feature = txt_session.run(["unnorm_text_features"], {"text":one_text})[0] # 未归一化的文本特征
    text_feature = torch.tensor(text_feature)
    text_features.append(text_feature)
text_features = torch.squeeze(torch.stack(text_features),dim=1) # 4个特征向量stack到一起
text_features = text_features / text_features.norm(dim=1, keepdim=True) # 归一化后的Chinese-CLIP文本特征，用于下游任务
print(text_features.shape) # Torch Tensor shape: [4, 特征向量维度]


