import csv
import gc
import os
from pathlib import Path
import time
from PIL import Image
import torch
import gradio as gr
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, FuyuImageProcessor, FuyuProcessor, FuyuForCausalLM
from PIL import Image
import numpy as np

# Set the model-related global variable
model_path = "./model/"
stop_batch_processing = False
model_loaded = False
# check runtime type
if torch.cuda.is_available():
  model_id=model_path+"fuyu-8b-sharded"
  print(f"\nUsing the sharded model '{model_id}' with to GPU usage.\n")
  print("This model is faster due to 4bit quantization and GPU computation. 🚀\n")
else:
  model_id=model_path+"fuyu-8b"
  print(f"\nUsing the original model '{model_id}' without GPU usage.\n")
  print("This model, as of today, can't be 4bit quantized. Also the weights can't be fully loaded into memory in colab free tier.")
  print("Running it on CPU and without the above optimizations makes it extremely slow in computation, however you can still do it and run this 8 billion parameter model!\n")

class Fuyu():
    """Pretrained fuyu model of Adept via huggingface"""

    def __init__(self, model_id):
        # check if GPU can be used
        if torch.cuda.is_available():
            print("You are running the model on GPU.")
            self.device = torch.device("cuda")
            self.dtype = torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype
            )
            self.model = FuyuForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
        else:
            print("You are running the model on CPU, the runtime might be very slow. 🐌")
            self.device = torch.device("cpu")
            self.dtype = torch.bfloat16
            # 4bit quantization is currently not working with the latest version of transformers (as of today: 4.35.0.dev0), it is working with transformers 4.30, however fuyu is not integrated there.
            self.model = FuyuForCausalLM.from_pretrained(model_id, device_map=self.device, torch_dtype=self.dtype)

        # initialize tokenizer and fuyu processor, pretrained and via huggingface
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.processor = FuyuProcessor(image_processor=FuyuImageProcessor(), tokenizer=self.tokenizer)

    def prompt(self, text, image=None, out_tokens=50):
        """Prompt the model with a text and optional an image prompt."""

        if image is None:
            # if no image is provided, use a small black image
            # Warning: This is working but the model is not trained on this image fake. Test purpose only!
            image = np.zeros((30,30,3), dtype=np.uint8)

        # pre processing image and text
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[-1]

        # process
        t0 = time.time()
        generation_output = self.model.generate(**inputs, max_new_tokens=out_tokens, pad_token_id=self.tokenizer.eos_token_id)
        print(f"\nGeneration time: {time.time()-t0:.0f}s")

        # post processing
        generation_text = self.tokenizer.decode(generation_output[0][prompt_len:], skip_special_tokens=True)
        return generation_text.lstrip()

Fuyu = None
def load_model():
    global Fuyu, model_loaded
    if not model_loaded:
        Fuyu.__init__
        model_loaded = True
        print("Model loaded.")
    else:
        print("Model already loaded.")


def unload_model():
    global Fuyu, model_loaded
    if model_loaded:
        Fuyu = None
        model_loaded = False
        torch.cuda.empty_cache()
        del Fuyu
        gc.collect()
        print("Model unloaded.")
    else:
        print("Model already unloaded.")

def gen_caption(image, prompt, max_new_tokens):
    load_model()
    if Fuyu is None:
        print("Model not loaded.")
        return None
    generation_text = Fuyu.prompt(image, prompt, max_new_tokens)
    return generation_text

def save_csv_f(caption, output_dir, image_filename):
    type = 'a' if os.path.exists(f'{output_dir}/blip2_caption.csv') else 'x'
    with open(f'{output_dir}/blip2_caption.csv', type, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        csvlist = [image_filename]
        csvlist.extend(caption.splitlines())
        writer.writerow(csvlist)


def save_txt_f(caption, output_dir, image_filename):
    if os.path.exists(f'{output_dir}/{os.path.splitext(image_filename)[0]}.txt'):
        f = open(f'{output_dir}/{os.path.splitext(image_filename)[0]}.txt', 'w', encoding='utf-8')
    else:
        f = open(f'{output_dir}/{os.path.splitext(image_filename)[0]}.txt', 'x', encoding='utf-8')
    f.write(f'{caption}\n')
    f.close()
        
        

def prepare(image, process_type, input_dir, output_dir, save_csv, save_txt, prompt, max_length):
    if not model_loaded:
        return "Model not loaded."
    if process_type == "Single Image":
        return gen_caption(image, prompt, max_length)
    elif process_type == "Batch Process":
        # Validate directories
        input_dir_path = Path(input_dir).resolve()
        output_dir_path = Path(output_dir).resolve()
        if not input_dir_path.is_dir():
            return "Input directory does not exist"
        if not output_dir_path.is_dir():
            output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get a list of images
        image_files = [
            f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        # Initialize tqdm progress bar
        progress_bar = tqdm(total=len(image_files), desc='Processing Images', unit='img')

        for image_filename in image_files:
            global stop_batch_processing
            if stop_batch_processing:
                stop_batch_processing = False
                return "Batch processing is stopped."
            image = Image.open(f"{input_dir}/{image_filename}")
            print(f"Processing {image_filename}")
            print("Caption:")
            caption = gen_caption(image, prompt, max_length)
            if save_csv:
                save_csv_f(caption, output_dir, image_filename)
            if save_txt:
                save_txt_f(caption, output_dir, image_filename)
            image.close()
            
            # Update the progress bar
            progress_bar.update(1)
        # Close the progress bar
        progress_bar.close()
        
        return f"Processed {len(image_files)} images!"
    
# A function to stop the batch processing
def stop_batch():
    global stop_batch_processing
    stop_batch_processing = True
    print("Batch processing stopped.")

# Define a separate function for single image processing
def prepare_single_image(input_image, max_length, prompt):
    print(f"Processing {input_image}")
    return prepare(input_image, "Single Image", None, None, False, False,
                   prompt, max_length)

# and another for batch processing
def prepare_batch(input_dir, output_dir, save_csv, save_txt, prompt, max_length):
    return prepare(None, "Batch Process", input_dir, output_dir, save_csv, save_txt,
                   prompt, max_length)

# Main Gradio interface
def gui():
    with gr.Blocks(title="Fuyu-8B Caption Generator") as demo:
        global model_path, parameter
        gr.Markdown("# Fuyu-8B Caption Generator")
        with gr.Row():
            with gr.Row():
                load_model_button = gr.Button("Load Model")
                unload_model_button = gr.Button("Unload Model")
                # save_model_path_button = gr.Button("Save Model Path")
            # model_path_input = gr.Textbox(label="Model Path", value=model_path)
            # save_model_path_button.click(save_model_path, inputs=model_path_input, outputs=None)
            # quantization_dropdown = gr.Dropdown(choices=["", "8bit", "4bit"], label="Quantization")
            load_model_button.click(load_model, inputs=None, outputs=None)
            unload_model_button.click(unload_model, inputs=None, outputs=None)
        max_length_slider = gr.Slider(minimum=5, maximum=100, step=1, label="Max Length", value=75)
        prompt = gr.Textbox(label="Prompt", placeholder="Enter your description prompt here.")
        with gr.Tabs() as tabs:
            with gr.TabItem("Single Image"):
                input_image = gr.Image(label="Image", type='filepath')
                output_text_single = gr.Textbox(label="Generated Caption(s)", lines=10, placeholder="Generated captions will appear here...")
                # prompt = gr.Textbox(label="Prompt", placeholder="Enter your description prompt here.")
                generate_caption_btn = gr.Button("Generate Caption", variant="primary")
                generate_caption_btn.click(prepare_single_image, inputs=[input_image, max_length_slider, prompt], outputs=output_text_single)

            with gr.TabItem("Batch Process"):
                input_dir = gr.Textbox(label="Input Directory", placeholder="Enter the directory path...")
                output_dir = gr.Textbox(label="Output Directory", placeholder="Enter the output directory path...")
                save_csv = gr.Checkbox(label="Save as CSV", value=True)
                save_txt = gr.Checkbox(label="Save as TXT", value=False)
                # prompt = gr.Textbox(label="Prompt", placeholder="Enter your description prompt here.")
                output_text_batch = gr.Textbox(label="Batch Process Status", lines=10, placeholder="Batch processing status will appear here...")
                batch_process_btn = gr.Button("Process Batch", variant="primary")
                batch_process_btn.click(prepare_batch, inputs=[input_dir, output_dir, save_csv, save_txt, prompt, max_length_slider], outputs=output_text_batch)
                stop_process_btn = gr.Button("Stop Processing", variant="secondary")
                stop_process_btn.click(stop_batch, inputs=[], outputs=[])

        # Removed the Accordion block because there's no longer a switch between beam search and nucleus sampling
    return demo

if __name__ == "__main__":
    # Run the GUI
    app = gui()
    app.launch()