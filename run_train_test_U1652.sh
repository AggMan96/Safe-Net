name="SafeNet-block4-lr0.01-sp2-bs8-ep120-s256-U1652"
data_dir="./data/University-Release/train"
test_dir="./data/University-Release/test"
gpu_ids=0
num_worker=4
lr=0.01
sample_num=2
block=4
batchsize=8
triplet_loss=0
num_epochs=120
pad=0
views=2
h=256
w=256

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --num_worker $num_worker --views $views --lr $lr \
--sample_num $sample_num --block $block --batchsize $batchsize --triplet_loss $triplet_loss --num_epochs $num_epochs --h $h --w $w\

cd checkpoints/$name
for((i=119;i<=$num_epochs;i+=20));
do
  for((p = 0;p<=$pad;p+=10));
  do
    for ((j = 1; j < 3; j++));
    do
        python test.py --test_dir $test_dir --checkpoint net_$i.pth --mode $j --gpu_ids $gpu_ids --num_worker $num_worker --pad $pad --h $h --w $w
    done
  done
done