r"""FitLN on a decoder-only LM: trains only the normalization scales, everything else
frozen, no task head.

Usage:
    python -u decoder/fitln_decoder.py --model Qwen/Qwen2.5-1.5B --task sst2 \
        --lr 1.3e-3 --batch_size 8 --num_epoch 5 --seed 1
"""
import argparse
import os

import torch
from transformers import Trainer

import common
import norm_freeze
import run_common

FITLN_WEIGHTS_FILE = "fitln_decoder_weights.bin"


class FitLNDecoderTrainer(Trainer):
    """Saves only the unfrozen parameters, so a checkpoint is the task vector itself."""

    def _trainable_state_dict(self, full_state_dict=None):
        trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
        sd = full_state_dict if full_state_dict is not None else self.model.state_dict()
        return {k: v for k, v in sd.items() if k in trainable}

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self._trainable_state_dict(state_dict), os.path.join(output_dir, FITLN_WEIGHTS_FILE))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

    def _load_best_model(self):
        if self.state.best_model_checkpoint is None:
            return
        weights_path = os.path.join(self.state.best_model_checkpoint, FITLN_WEIGHTS_FILE)
        if not os.path.exists(weights_path):
            return
        partial = torch.load(weights_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(partial, strict=False)
        if unexpected:
            print(f"[FitLNDecoderTrainer] unexpected keys when loading best model: {unexpected}")


def save_compact_checkpoint(model, output_dir: str) -> str:
    final_weights = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    path = os.path.join(output_dir, FITLN_WEIGHTS_FILE)
    torch.save(final_weights, path)
    print(f"[fitln_decoder] Saved {len(final_weights)} trainable tensors to {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser("fitln_decoder")
    common.add_common_decoder_args(parser, default_lr=8e-4)
    parser.add_argument("--include_final_norm", type=common.str2bool, default=True,
                         help="Also unfreeze model.norm.weight (the final pre-LM-head RMSNorm), "
                              "as on the encoder side.")
    args = parser.parse_args()

    run_common.set_seed(args.seed)
    print("Arguments parsed\n\n")

    tokenizer = run_common.load_tokenizer(args.model)
    model = run_common.load_causal_lm(args.model)
    run_common.log_model_facts(model, args.model)

    norm_freeze.freeze_for_fitln_decoder(model, include_final_norm=args.include_final_norm)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print("Model loaded, frozen, and initialized\n\n")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    run_common.run_training(
        args,
        method="fitln",
        model=model,
        tokenizer=tokenizer,
        trainer_cls=FitLNDecoderTrainer,
        extra_trainer_kwargs={"optimizers": (optimizer, None)},
        compact_save_fn=save_compact_checkpoint,
    )
