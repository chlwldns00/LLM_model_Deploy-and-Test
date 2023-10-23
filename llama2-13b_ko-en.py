from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch

model_name="hyunseoki/ko-en-llama2-13b"
tokenizer = AutoTokenizer.from_pretrained("hyunseoki/ko-en-llama2-13b")
language_model = AutoModelForCausalLM.from_pretrained("hyunseoki/ko-en-llama2-13b")

model_save_path='/workspace/ko-en_translation'
tokenizer.save_pretrained(model_save_path)
language_model.save_pretrained(model_save_path)


text_generation_pipeline = TextGenerationPipeline(
    model=language_model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device=0,
)


def generate_translation(input_text):
    output = text_generation_pipeline(input_text, max_length=50)  # 적절한 길이로 조정
    return output[0]['generated_text']


iface = gr.Interface(
    fn=generate_translation,
    inputs="text",
    outputs="text",
    title="Korean to English Translation",
    description="Translate Korean text to English.",
)

iface.launch()