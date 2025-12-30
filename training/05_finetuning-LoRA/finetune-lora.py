import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from peft import PeftModel

if len(sys.argv) >= 2:
    N_SAMPLES = int(sys.argv[1])
else:
    N_SAMPLES = 500

base_dir_name = "qwen2-7b-llmjp-lora-"
lora_output_dir = base_dir_name + str(N_SAMPLES)
lora_save_dir = lora_output_dir + "-save"
lora_merged_dir = lora_output_dir + "-merged"

SYSTEM = "あなたは日本語の有能なアシスタントです。"
MAX_LEN = 2048

def preprocess_example(example):
    instr = (example.get("instruction") or "").strip()
    inp   = (example.get("input") or "").strip()
    out   = (example.get("output") or "").strip()

    ng_result = {"input_ids": [], "labels": [], "keep": False}

    if (not instr and not inp) or (not out):
        return ng_result

    user_context = f"{instr}\n\n入力:\n{inp}" if inp else instr
    prompt_msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_context},
    ]
    full_msgs = prompt_msgs + [{"role": "assistant", "content": out}]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        full_msgs, tokenize=True, add_generation_prompt=False
    )

    if len(full_ids) <= len(prompt_ids):
        return ng_result

    if len(full_ids) > MAX_LEN:
        full_ids = full_ids[:MAX_LEN]
        if len(full_ids) <= len(prompt_ids):
            return ng_result

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels, "keep": True}

def collate_fn(batch):
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    labels    = [torch.tensor(x["labels"],    dtype=torch.long) for x in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    attention_mask = (input_ids_padded != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded,
    }

model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast = True,
)

# The reason why pad_token is checked is that some LLMs does not use
# pad_token specifically, but eos_token.
# For the case of Qwen2, pad_token is <|endoftext|>
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype = "auto",
#    device_map = "auto"
)

model.config.pad_token_id = tokenizer.pad_token_id

raw_dataset = load_dataset("izumi-lab/llm-japanese-dataset", split="train")
train_dataset = raw_dataset.shuffle(seed=42).select(range(N_SAMPLES))

processed = train_dataset.map(
    preprocess_example,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing",
)
processed = processed.filter(lambda x: x["keep"])
processed = processed.remove_columns(["keep"])

lora_config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

lora_model = get_peft_model(model, lora_config)
lora_model.enable_input_require_grads()
lora_model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir = lora_output_dir,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    num_train_epochs = 1,
    learning_rate = 2e-4,
    warmup_ratio = 0.03,
    max_grad_norm = 0.3,
    group_by_length = True,
    logging_steps = 50,
    save_steps = 1000,
    save_total_limit = 2,
    bf16 = True,
#    optim = "paged_adamw_32bit",
    optim = "adamw_torch",
    gradient_checkpointing = True,
    dataloader_num_workers = 0,
    report_to = [],
)

trainer = Trainer(
    model = lora_model,
    args = training_args,
    train_dataset = processed,
    data_collator = collate_fn,
    processing_class = tokenizer,
)
lora_model.config.use_cache = False

result = trainer.train()
trainer.save_state()

print("Training finished.")
print(result.metrics)

save_dir = lora_save_dir
lora_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

merged_dir = lora_merged_dir

peft_model = PeftModel.from_pretrained(model, save_dir)
merged = peft_model.merge_and_unload()

merged.save_pretrained(merged_dir, safe_serialization=True)
