# Fuyu-8B-captioner

A simple tool to generate image captions using the Fuyu-8B model.

Requirements:
- Python 3.10
- PyTorch
- A GPU with at least 16GB of memory if using CUDA
  
Installation:

- Use a environment with Python 3.10 and PyTorch installed.
- Clone the repository: `git clone https://github.com/jiayev/Fuyu-8B-captioner.git`
- `cd Fuyu-8B-captioner`
- Install dependencies: `pip install -r requirements.txt`

Download the pre-trained model from the link below and place it in ./model/

https://huggingface.co/ybelkada/fuyu-8b-sharded

It will look like this:
```
├── model
│   └── fuyu-8b-sharded
```

Or download from https://huggingface.co/adept/fuyu-8b if you use CPU or TPU.

To run the tool, use the following command:
```
python main.py
```
