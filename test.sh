env CUDA_VISIBLE_DEVICES=0 python -u train.py --share_vocab --save_model tmp_model.pt --test  --title
perl detokenizer.perl -l en < tmp_gold.txt > tmp_gold.txt.a
perl detokenizer.perl -l en < tmp_pred.txt > tmp_pred.txt.a
perl multi-bleu.perl tmp_gold.txt < tmp_pred.txt
perl multi-bleu-detok.perl tmp_gold.txt.a < tmp_pred.txt.a
