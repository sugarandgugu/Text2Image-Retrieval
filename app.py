import gradio as gr
from text2image import text2image_gr

if __name__ == "__main__":
    gr.close_all()
    with gr.TabbedInterface(
            [text2image_gr()],
            ["文到图搜索"],
    ) as demo:
        demo.launch()