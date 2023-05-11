

#1.创建conda环境:
conda create -n HiVT python=3.8
conda activate HiVT
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pytorch-geometric==1.7.2 -c rusty1s -c conda-forge
conda install pytorch-lightning==1.5.2 -c conda-forge


#2.保存模型参数，训练损失
mkdir ~/shangqi_train

#3.Training
python train_shangqi.py  --embed_dim 64

