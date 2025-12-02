python src\eval_ppl.py --model EleutherAI/pythia-70m --dataset wikitext --split test --stride 1024 --max_length 2048
python src\speed_benchmark.py --model EleutherAI/pythia-70m --dataset pg19 --mode dense --max_new_tokens 256
python src\speed_benchmark.py --model EleutherAI/pythia-70m --dataset pg19 --mode snapkv --compression_ratio 0.5 --max_new_tokens 256
python src\speed_benchmark.py --model EleutherAI/pythia-70m --dataset pg19 --mode pyramidkv --compression_ratio 0.7 --max_new_tokens 256
python src\speed_benchmark.py --model EleutherAI/pythia-70m --dataset pg19 --mode streaming --head_window 1024 --tail_window 2048 --max_new_tokens 256
