import argparse
from dino_peft.trainers.seg_trainer import SegTrainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg",
        "--config",
        dest="cfg",
        type=str,
        default="configs/mac/lucchi_lora_mac.yaml",
        help="Path to YAML config.",
    )
    args = ap.parse_args()

    trainer = SegTrainer(args.cfg)
    trainer.train()

if __name__ == "__main__":
    main()
