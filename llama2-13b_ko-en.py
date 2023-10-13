from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("hyunseoki/ko-en-llama2-13b")
model = AutoModelForCausalLM.from_pretrained("hyunseoki/ko-en-llama2-13b")

model=model()
tok=tokenizer()