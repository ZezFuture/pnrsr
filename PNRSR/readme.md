### 环境

mkdir -p /opt/conda/envs/NFT2
tar -xzf /data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188740/pkg/NFT.tar.gz -C /opt/conda/envs/NFT2
conda activate NFT2
pip install loguru
cd /data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188740/code/Z-Image-main
pip install -e .

如果因为机器原因无法执行上述命令。直接在torch大于2.5版本的环境执行以下命令
pip install diffusers==0.37.1
pip install transformers==5.5.4
pip install huggingface_hub==1.11.0
pip install peft==0.19.1
pip install loguru
pip install lpips
pip install FDL_pytorch
cd /data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188740/code/Z-Image-main
pip install -e .

若提示basicsr出错
参考https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985可修复
其他依赖包我不太记得了，遇到缺少什么再装什么吧




### 替换lr数据集

cd /data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188740/code/Z-Image-main/srmodel4_onestep_onlylora
修改infer_tiny.sh
--input_image
--output_dir 
其他不要动

### 运行
cd /data/vjuicefs_ai_camera_jgroup_acadmic/public_data/11188740/code/Z-Image-main
bash srmodel4_onestep_onlylora/infer_tiny.sh


