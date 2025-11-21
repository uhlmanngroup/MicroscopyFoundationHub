import argparse
from dino_peft.trainers.seg_trainer import SegTrainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config/lucchi_lora_mac.yaml")
    args = ap.parse_args()

    trainer = SegTrainer(args.cfg)
    trainer.train()

if __name__ == "__main__":
    main()
