from transformers import AutoProcessor,AutoModelForImageTextToText

#加载预训练模型
model_path = "model/Qwen3-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model,output_loading_info = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    output_loading_info=True,
)

message = [
    {
        "role":"user",
        "content":[
            {
                "type":"video",
                "video":"src/storage/dog_run.mp4"
            },

            {
                "type":"text",
                "text":"描述一下这个视频"
            }
        ],
    }
]
inputs = processor.apply_chat_template(
    message,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs,max_new_tokens = 128)
# 从生成的完整 token ID 序列中剔除输入部分的 token，仅保留模型新生成的部分
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
# 将新生成的 token ID 解码为可读的文本字符串
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)


