export DEBUG_SPEED=True
export WORLD_SIZE=1

#export CUDA_VISIBLE_DEVICES=1
export TOT_SWAP_BYTES=5368709120
export SWAP_TO_GPU=0
export ASYNC_COMPRESS=1

export CUDA_VISIBLE_DEVICES=1
#nsys profile -t cuda,osrt,nvtx,cublas,cudnn --gpuctxsw=true -y 10 -d 30 -o actnn-async-compress-resnet50-b128 -f true -w true \

python train.py \
        --data /disk2/zhaojp/imagenet/imagenetfolder/ \
	-a resnet50 \
	--batch-size 464 \
	--alg L1 \
	--benchmark gact

# T4 L1 max: 464: 480 OOM due to mem-frag:
#Tried to allocate 1.44 GiB (GPU 0; 14.76 GiB total capacity; 7.94 GiB already allocated; 1.40 GiB free; 12.07 GiB reserved in total by PyTorch)
#	--alg L1 \
#--batch-size 64 --alg L1 --benchmark actnn
