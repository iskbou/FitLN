# FitLN

FitLN fine-tunes only the post-attention normalization layer of each transformer block
(plus the final norm) and keeps everything else frozen. A trained task is a small vector of
normalization weights, so one frozen backbone can serve many tasks.

This repository contains the encoder experiments (BERT, RoBERTa, DeBERTa-v3, ModernBERT;
GLUE and BEIR), the decoder-only experiments (Llama-3.2, Qwen2.5, Qwen3.5; FitLN, LoRA and
full fine-tuning), the task-vector analyses and a benchmark of task capacity, task switching
and training speed.

## Layout

- encoder/: GLUE and BEIR training, aggregation and analysis scripts
- decoder/: decoder training scripts, aggregation and analysis
- hotswap/: capacity, task-switching and throughput benchmark
- analysis/: task-vector utilities shared by the analysis scripts
- notebooks/: decoder task-vector analysis notebook
- results/encoder/, results/decoder/, results/hotswap/: result tables, cached parameters and significance tests
- figures/encoder/, figures/decoder_task_vector/, figures/hotswap/: figures

## Environment

    pip install -r requirements.txt

BEIR training and evaluation need sentence-transformers and mteb (mteb 2.x); the violin plots
need seaborn 0.13 or later. The Qwen3.5 models need Python 3.10 or later and a transformers
release with qwen3_5 support (5.x). Llama-3.2 and Qwen2.5 run on transformers 4.46 or later.
Llama-3.2 is gated on the Hugging Face Hub (accept the licence, then run huggingface-cli login).
Training needs a GPU.
All commands are run from the repository root.

# Encoders

## GLUE

The training scripts read GLUE from the Hugging Face hub. Models: bert-large-uncased,
FacebookAI/roberta-large, microsoft/deberta-v3-large, answerdotai/ModernBERT-large. Tasks:
cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte. The training set is split 85/15 into train and
validation (fixed seed 0) and the GLUE validation set is used as test set. The FitLN runs
are single seed (seed 0).

FitLN trains the attention-output LayerNorm of every layer (the attn_norm of ModernBERT, which
also trains its final norm) and the classifier. The pooler is frozen unless --train_pooler is
given. The learning rate is constant, without warm-up; --lr_schedule cosine gives a cosine
schedule with warm-up.

    python -u encoder/fitln_glue.py --model FacebookAI/roberta-large --task rte \
        --lr 1e-3 --batch_size 16 --num_epoch 10

Results are written to out/{task}/{model}/attn_only/{lr}/{num_epoch}/: fitln_weights.bin and
results.json (results_m.json and results_mm.json for mnli, matched and mismatched).

BitFit and LoRA are run by one script. Splits and learning rate are as above, except that the
training set is split 90/10.

    python -u encoder/peft_glue.py --method bitfit --model FacebookAI/roberta-large \
        --task rte --lr 5e-4 --batch_size 16 --num_epoch 10

    python -u encoder/peft_glue.py --method lora --lora_r 1 --lora_alpha 8 \
        --model FacebookAI/roberta-large --task rte --lr 5e-4 --batch_size 16 --num_epoch 10

    python -u encoder/peft_glue.py --method lora --lora_r 8 --lora_alpha 32 \
        --model FacebookAI/roberta-large --task rte --lr 5e-4 --batch_size 16 --num_epoch 10

Results are written to out/bitfit/ or out/lora_r{r}/, then {model}/{task}/{lr}/{seed}/
(val_results.json, test_results.json, bitfit_weights.bin or adapter/). For ModernBERT the LoRA
rank is three times --lora_r.

Grids of the reported results:

- FitLN: learning rates 1e-4 to 10e-4 in steps of 1e-4, 10 epochs, batch size 16. For qqp and
  mnli: learning rates 1e-4, 4e-4, 6e-4, 8e-4, 10e-4 and batch size 24.
- BitFit and LoRA: learning rates 5e-5, 1e-4, 5e-4, 8e-4, 10 epochs, batch size 16.
- mrpc, stsb and rte can start from the MNLI run: add --mnli_init followed by
  out/mnli/{model}/attn_only/{lr}/10/fitln_weights.bin (FitLN),
  out/lora_r{r}/{model}/mnli/{lr}/0/adapter (LoRA) or
  out/bitfit/{model}/mnli/{lr}/0/bitfit_weights.bin (BitFit). The classifier is not loaded.

The learning rate is selected by the mean validation score over runs. The test score of that
learning rate is reported. For FitLN the runs of the different epoch directories are averaged
like seeds. cola uses the Matthews correlation, stsb the Pearson correlation and the other
tasks accuracy. The test score of mnli is the mean of matched and mismatched accuracy for
BitFit and LoRA, and the mismatched accuracy for FitLN.

    python encoder/aggregate_glue.py --root out --peft_root out --out results/encoder/glue_detail.csv --table
    python encoder/aggregate_glue.py --csv_in results/encoder/glue_detail.csv --table

The second command prints the table from the shipped results/encoder/glue_detail.csv. The full
fine-tuning scores of the table are taken from Warner et al. (2025) and are in
results/encoder/fullft_glue_borrowed.csv; there is no training script for them here.

Notes on the baselines:

- For BERT the LoRA run labelled rank 1 used rank 8 and alpha 8 (--lora_r 8 --lora_alpha 8).
- The LoRA runs of rank 8 used alpha 32.
- Raw runs of LoRA rank 1 are missing for some tasks, so those cells are in the shipped CSV only.

## BEIR

Training on MS MARCO triplets with hard negatives (1.25M examples, batch size 512, cached
in-batch negatives loss, 1 epoch, warm-up 5 percent), then evaluation on 15 BEIR tasks with
mteb (NDCG@10). Trained: attention-output LayerNorms, final norm, classifier and pooler.

    python -u encoder/beir_train.py --model FacebookAI/roberta-large --lr 1.2e-3
    python -u encoder/beir_eval.py --model FacebookAI/roberta-large --lr 1.2e-3

The model is saved to out_ri/msmarco/{model}/{lr}/1/{model_short}-DPR-{lr}/final and the mteb
scores to result_mteb/mteb. With --task one task is evaluated. The learning rate is selected
among 3e-4 to 12e-4 by the mean NDCG@10 on NFCorpus, SciFact, TRECCOVID and FiQA2018 (run
beir_eval.py with --cache_dir set to a separate directory for these runs):

    python encoder/aggregate_beir.py select --dir result_mteb_for_selection/mteb/roberta-large
    python encoder/aggregate_beir.py table --dir result_mteb/mteb --dir_star result_mteb_zs/mteb \
        --lrs results/encoder/beir_lrs.json --fullft results/encoder/beir_fullft.csv \
        --out results/encoder/beir_final_results.csv

FitLN uses the selected learning rate and FitLN* the fixed learning rate of beir_lrs.json. The
table gives the NDCG@10 per task, the ratio to full fine-tuning and the meta-task averages.

## Task-vector analysis

Task vectors are the trained minus pretrained LayerNorm weights and biases. The scripts read
the shipped results/encoder/ln_params.pt, so they need neither the raw runs nor the models.

    python encoder/analysis_task_vectors.py figures --cache results/encoder/ln_params.pt --figdir figures/encoder
    python encoder/analysis_task_vectors.py pruning_plot --csv results/encoder/pruning_curves.csv --figdir figures/encoder
    python encoder/analysis_violins.py plot --cache results/encoder/roberta_violin_params.pt --figdir figures/encoder

To rebuild the cached files from the runs in out/ (the pretrained models are downloaded):

    python encoder/analysis_task_vectors.py cache --root out --out results/encoder/ln_params.pt
    python encoder/analysis_violins.py cache --root out --out results/encoder/roberta_violin_params.pt

The figures are the cosine similarity between tasks, the sparsity and magnitude of the task
vectors per layer, the cross-model comparison, the pruning curves and the violin plots of the
LayerNorm parameters of RoBERTa-large. MNLI is not in these figures. The pruning evaluation
(analysis_task_vectors.py pruning --root out) needs a GPU. The violin plot of the MS MARCO model
is made when a checkpoint saved by beir_train.py is given to the cache command with
--msmarco_ckpt.

# Decoders

## Training

One run is one python command. Models: meta-llama/Llama-3.2-1B, meta-llama/Llama-3.2-3B,
Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-3B, Qwen/Qwen3.5-2B-Base, Qwen/Qwen3.5-4B-Base.
Tasks: sst2, rte, mrpc, cola (GLUE as generation, scored by label log-likelihood) and
alpaca_mini, gsm8k, samsum (free-form generation; ROUGE-L, exact match, ROUGE-L).
alpaca_mini is the 2,002-example held-out split of the 52K Stanford Alpaca set, not AlpacaEval.

    python -u decoder/fitln_decoder.py --model Qwen/Qwen2.5-1.5B --task sst2 \
        --lr 1.3e-3 --batch_size 8 --num_epoch 5 --seed 1

    python -u decoder/lora_decoder.py --model Qwen/Qwen2.5-1.5B --task sst2 \
        --lr 2e-4 --lora_r 8 --batch_size 8 --num_epoch 3 --seed 1

    python -u decoder/fullft_decoder.py --model Qwen/Qwen2.5-1.5B --task sst2 \
        --lr 2e-5 --batch_size 8 --num_epoch 3 --seed 1

For gsm8k add --max_new_tokens 256.

Settings of the reported results:

- FitLN: learning rates 8e-4, 9e-4, 10e-4, 11e-4, 12e-4, 13e-4; 5 epochs. Trains
  post_attention_layernorm of every layer and the final norm (scale only).
- LoRA: rank 8 and 1; learning rates 5e-5, 1e-4, 2e-4; 3 epochs; alpha = 2 * rank;
  dropout 0.05; q/k/v/o_proj (Qwen3.5 also the linear-attention in_proj_* and out_proj).
- Full fine-tuning: learning rates 1e-5, 2e-5; 3 epochs; bf16 weights.
- All methods: batch size 8, maximum sequence length 256, constant learning rate without
  warm-up, bf16 autocast, evaluation and checkpoint every 200 steps with early stopping
  (patience 8) on held-out loss, greedy decoding for free-form tasks, seeds 1, 2 and 42.

A run writes to out/decoder/{method}/{model}/{task}/{lr}/{seed}/: val_results.json,
test_results.json, train_meta.json (written last), and fitln_decoder_weights.bin (FitLN) or
adapter/ (LoRA).

## Full sweep

For one model (repeat for each of the six models). Runs are independent.

    MODEL=Qwen/Qwen2.5-1.5B
    TASKS="sst2 rte mrpc cola alpaca_mini gsm8k samsum"
    EXTRA() { [ "$1" = gsm8k ] && echo "--max_new_tokens 256"; }

    for seed in 1 2 42; do for task in $TASKS; do for lr in 8e-4 9e-4 10e-4 11e-4 12e-4 13e-4; do
      python -u decoder/fitln_decoder.py --model $MODEL --task $task --lr $lr \
          --batch_size 8 --num_epoch 5 --seed $seed $(EXTRA $task)
    done; done; done

    for seed in 1 2 42; do for task in $TASKS; do for r in 8 1; do for lr in 5e-5 1e-4 2e-4; do
      python -u decoder/lora_decoder.py --model $MODEL --task $task --lora_r $r --lr $lr \
          --batch_size 8 --num_epoch 3 --seed $seed $(EXTRA $task)
    done; done; done; done

    for seed in 1 2 42; do for task in $TASKS; do for lr in 1e-5 2e-5; do
      python -u decoder/fullft_decoder.py --model $MODEL --task $task --lr $lr \
          --batch_size 8 --num_epoch 3 --seed $seed $(EXTRA $task)
    done; done; done

The full sweep is large, so the aggregated results are in results/decoder/.

## Aggregation

The learning rate is chosen once per (method, model, task) by the lowest mean validation
loss across seeds. Reported values are the mean and sample std over the seed runs at that
learning rate. Seeds 1, 2 and 42 are used by default; --seeds all uses every seed found.

    python decoder/aggregate_decoder_results.py --root out/decoder --out_dir results/decoder
    python decoder/summarize_results.py --root out/decoder --out results/decoder/global_summary.md
    python decoder/significance_testing/generate_significance_tests.py

These two read results/decoder/global_summary.md and need no raw runs:

    python decoder/analyze_training_memory.py --summary_md results/decoder/global_summary.md --out_dir results/decoder
    python decoder/reasoning_gap_correlation.py --summary_md results/decoder/global_summary.md --out_dir results/decoder

## Task vectors

notebooks/task_vector_analysis_decoder.ipynb computes the task vectors (fine-tuned minus
pretrained normalization weights), their cosine similarity within and across tasks, and the
seed stability of the salient channels. It needs the trained fitln_decoder_weights.bin files
under out/decoder/fitln/ and the pretrained backbones. Its tables are in
results/decoder/task_vector/ and its figures in figures/decoder_task_vector/.

## Notes

RMSNorm has a scale but no bias, so the decoder variant is scale-only. In Llama and Qwen,
post_attention_layernorm is the norm before the MLP, the closest counterpart of the
encoder's attention-output LayerNorm but not equivalent. The GLUE tasks use the frozen
language-model head and score label tokens by log-likelihood, so no task head is trained.

# Hot-swap benchmark

Three measurements for FitLN, BitFit, LoRA (ranks 1 and 8) and full fine-tuning (BitFit on
encoders only), for the four encoders and six decoders:

- capacity: how many task-specific parameter sets fit on one GPU next to one frozen backbone
  (arithmetic from the parameter counts, no GPU needed);
- task switching: time to make an already resident task active (resident), to load a task
  from host memory (cold) and, for LoRA, to merge the adapter into the base weights;
- throughput: inference and training-step samples per second, on the decoders.

Model names are those of hotswap/registry.py (for example llama-3.2-1b, qwen2.5-3b,
bert-large, modernbert-large). The encoder capacity counts a classifier and pooler head of
1,051,650 parameters per task for every method (2,050 for ModernBERT). --include_head false
counts the deltas alone.

    python hotswap/capacity.py --dtype bf16 --reserve_frac 0.10 --out_dir results/hotswap
    python hotswap/capacity.py --include_head false --out_dir results/hotswap

One switching run is one command; it writes out/hotswap/{arch}/{model}/{method}/
hotswap_result.json. The reported runs used an H100 with 300 switches and 8 resident tasks.

    python -u hotswap/run_hotswap.py --arch decoder --model llama-3.2-1b --method fitln \
        --device cuda --n_resident 8 --n_switches 300

    for method in fitln lora1 lora8 fullft; do
      for model in llama-3.2-1b llama-3.2-3b qwen2.5-1.5b qwen2.5-3b qwen3.5-2b-base qwen3.5-4b-base; do
        python -u hotswap/run_hotswap.py --arch decoder --model $model --method $method \
            --device cuda --n_resident 8 --n_switches 300
      done
    done

    for method in fitln bitfit lora1 lora8 fullft; do
      for model in bert-large roberta-large deberta-v3-large modernbert-large; do
        python -u hotswap/run_hotswap.py --arch encoder --model $model --method $method \
            --device cuda --n_resident 8 --n_switches 300
      done
    done

qwen3.5-4b-base with fullft was run with --n_resident 4 (out of memory at 8). Thirteen of the
44 cells load the real model with --real_model True instead of synthetic tensors: fitln and
fullft on llama-3.2-1b and qwen2.5-1.5b, and fitln, bitfit and fullft on bert-large,
deberta-v3-large and modernbert-large. The real-model encoder fitln delta contains the
attention-output LayerNorm, the classifier and the pooler. LoRA is not implemented for
--real_model True. The latency table and the pairwise tests (Welch's t-test on the cold swap,
Benjamini-Hochberg correction) are built from the runs; the shipped runs are in
results/hotswap/runs/.

    python hotswap/aggregate_hotswap_results.py --out_root results/hotswap/runs --results_dir results/hotswap
    python hotswap/plot_capacity_swap.py --results_dir results/hotswap --runs results/hotswap/runs --figdir figures/hotswap

Throughput is measured one model at a time, alone on the GPU, with batch size 8, sequence
length 256 and the same dtype for every method (the reported results use fp16), 10 warm-up
and 50 timed iterations. LoRA inference is also timed with the adapter merged.

    python -u hotswap/throughput_benchmark.py --model meta-llama/Llama-3.2-1B --dtype fp16 \
        --methods fitln fullft lora1 lora8 --batch_size 8 --seq_len 256 --n_warmup 10 --n_iters 50
    python hotswap/plot_throughput.py --results_dir results/hotswap --figdir figures/hotswap

The same command is run for meta-llama/Llama-3.2-3B, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-3B,
Qwen/Qwen3.5-2B-Base and Qwen/Qwen3.5-4B-Base. Outputs are results/hotswap/throughput_*.json
and the summary results/hotswap/throughput_summary.md with the speed-up ratios.

# License

MIT, see LICENSE.
