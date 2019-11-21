env CUDA_VISIBLE_DEVICES=1 python -u train.py --share_vocab --prop 2 --save_model tmp_model.pt --title --lr 1e-3 > train_1.log 2>&1 &
