from transformers import pipeline
import gradio as gr

pipe = pipeline("text2text-generation", model="inhee/opus-mt-ko-en-finetuned-ko-to-en5")

demo = gr.Interface.from_pipeline(pipe)
demo.launch()
