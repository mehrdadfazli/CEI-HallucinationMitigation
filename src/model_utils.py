import torch
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    LlavaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    BitsAndBytesConfig,
    AutoProcessor
)
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

model_names = {
    "instructblip": "Salesforce/instructblip-vicuna-7b",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava-next": "llava-hf/llava-v1.6-vicuna-7b-hf"
}

def load_model_and_processor(model_type, model_names, cache_dir, device, load_in_8bit=True):
    """
    Load the model and processor based on the model type.
    """
    try:
        quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit, llm_int8_threshold=200.0, bnb_4bit_compute_dtype=torch.float16,  # ✅ this fixes the warning
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4") 

        if model_type == "instructblip":
            model = InstructBlipForConditionalGeneration.from_pretrained(
                model_names[model_type],
                torch_dtype=torch.float16,
                attn_implementation="eager",
                cache_dir=cache_dir,
                quantization_config=quantization_config,
                device_map="auto"
            )
            model.tie_weights()
            processor = InstructBlipProcessor.from_pretrained(model_names[model_type], cache_dir=cache_dir)
        elif model_type == "llava":
            model = LlavaForConditionalGeneration.from_pretrained(
                model_names[model_type],
                torch_dtype=torch.float16,
                attn_implementation="eager",
                cache_dir=cache_dir,
                quantization_config=quantization_config,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_names[model_type], cache_dir=cache_dir)
            processor.patch_size = model.config.vision_config.patch_size
            processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy


        elif model_type == "llava-next":
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_names[model_type],
                torch_dtype=torch.float16,
                attn_implementation="eager",
                cache_dir=cache_dir,
                quantization_config=quantization_config,
                device_map="auto"
            )
            processor = LlavaNextProcessor.from_pretrained(model_names[model_type], cache_dir=cache_dir)
            processor.patch_size = model.config.vision_config.patch_size
            processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

        elif model_type == "qwen2-vl":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_names[model_type],
                torch_dtype=torch.float16,
                attn_implementation="eager",
                device_map="auto",
                cache_dir=cache_dir
            )
            processor = AutoProcessor.from_pretrained(
                model_names[model_type],
                cache_dir=cache_dir
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        logger.info(f"Loaded {model_type} model and processor")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading model and processor: {e}")
        raise

def process_inputs(raw_image, query, processor, model_type, device="cuda"):
    """
    Process inputs depending on the model (e.g., InstructBLIP or LLaVA).
    """
    try:
        if model_type == "llava" or model_type == "llava-next":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=raw_image, text=text_prompt, padding=True, return_tensors="pt").to(device, torch.float16)
        
        elif model_type == "instructblip":
            inputs = processor(images=raw_image, text=query, return_tensors="pt").to(device)
        
        elif model_type == "qwen2-vl":
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": raw_image},
                        {"type": "text", "text": query}
                    ]
                }
            ]        
            
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text_prompt, images=raw_image, return_tensors="pt").to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return inputs
    except Exception as e:
        logger.error(f"Error processing inputs: {e}")
        raise
     