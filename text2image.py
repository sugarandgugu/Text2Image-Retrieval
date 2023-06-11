from functools import partial
import json
from multiprocessing.pool import ThreadPool as Pool
import gradio as gr
from utils import *


def text2image_gr():
    

    examples = [
        ["游泳的狗", 20, clip_base, "是"],
        ["夜晚盛开的荷花", 20, clip_base, "是"],
        ["一个走在公园里的女孩", 20, clip_base, "是"],
        ["抱着孩子的男人", 20, clip_base, "是"]
    ]

    title = "<h1 align='center'>中文CLIP文到图搜索应用</h1>"

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=2):
                    text = gr.Textbox(value="戴着眼镜的猫", label="请填写文本", elem_id=0, interactive=True)
                num = gr.components.Slider(minimum=0, maximum=50, step=1, value=8, label="返回图片数（可能被过滤部分）", elem_id=2)
                model = gr.components.Radio(label="模型选择", choices=[clip_base, clip_large, clip_large_336],
                                            value=clip_base, elem_id=3)
                thumbnail = gr.components.Radio(label="是否返回缩略图", choices=[yes, no],
                                                value=yes, elem_id=4)
                btn = gr.Button("搜索", )
            with gr.Column(scale=100):
                out = gr.Gallery(label="检索结果为：").style(grid=4, height=200)
        inputs = [text, num, model, thumbnail]
        btn.click(fn=clip_api, inputs=inputs, outputs=out)
        gr.Examples(examples, inputs=inputs)
    return demo


if __name__ == "__main__":
    with gr.TabbedInterface(
            [text2image_gr()],
            ["文到图搜索"],
    ) as demo:
        demo.launch(
            enable_queue=True,
)