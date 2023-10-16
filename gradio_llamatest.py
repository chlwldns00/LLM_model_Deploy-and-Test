from transformers import pipeline
import gradio as gr

pipe = pipeline("text2text-generation", model="hyunseoki/ko-en-llama2-13b")

demo = gr.Interface.from_pipeline(pipe)
demo.launch()
