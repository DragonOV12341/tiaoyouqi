# 自动调优器后端
用于第三方测试的交付   

入口点：stub_code/start.sh

环境配置：需要安装 torch-mlir 以及相关工具。建议在虚拟环境里配置 ：   
```shell
# python --version = 3.11.11
pip install --pre torch-mlir torchvision \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels

```