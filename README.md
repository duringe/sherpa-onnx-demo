# sherpa-onnx-demo

Basic demonstration from k2-fsa/sherpa-onnx project


来自k2-fsa/sherpa-onnx项目的中英文tts模型运行基础演示


## Installation and Running
### Steps
1. 进入文件夹运行venv环境
```bash
Scripts\activate
```
2. 运行run.py文件
```bash
python .\run.py
```
3. 在cmd或wt的窗口中根据input输入对应参数运行



## Building
### Steps
1. 创建venv环境
```bash
python -m venv your-porject-name
```
2. 下载依赖库
```bash
python -s -m pip install -r .\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
3. 下载模型相关文件,放到同目录中,再运行run.py
```bash
python .\run.py
```




## Credits
- sherpa-onnx - https://github.com/k2-fsa/sherpa-onnx
- onnx-model - https://modelscope.cn/models/QuadraV/tts-zh-en-onnx/summary
