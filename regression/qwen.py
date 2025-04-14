from transformers import AutoModelForCausalLM, AutoTokenizer

def qwen(model_name, prompt, no_context = True, content = "", messages_ = []):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = messages_
    if no_context:
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt}
        ]
    else:
        messages.append({"role": "user", "content": prompt})
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    messages.append({"role": "assistant", "content": response1})

    print(response1)

    # return messages

model_names = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]

prompt = "Расскажи мне на русском языке о машинном обучении."

qwen(model_names[0], prompt)
qwen(model_names[1], prompt)
qwen(model_names[2], prompt)