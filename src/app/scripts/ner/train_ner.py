import argparse
from datasets import load_dataset, concatenate_datasets
from src.app.core.ner.bert_ner import BertNER

def main():
    parser = argparse.ArgumentParser(description="Train BERT NER on jjzha/skillspan")
    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--output_dir", type=str, default="./models/bert-ner-skillspan")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_all_tokens", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--oversample_entities", type=int, default=1)
    args = parser.parse_args()

    dataset = load_dataset("jjzha/skillspan")
    train_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
    test_dataset = dataset["test"]

    train_texts = train_dataset["tokens"]
    train_labels = [[str(tag) for tag in tags] for tags in train_dataset["tags_skill"]]

    test_texts = test_dataset["tokens"]
    test_labels = [[str(tag) for tag in tags] for tags in test_dataset["tags_skill"]]

    ner_model = BertNER(model_name=args.model_name)
    trainer = ner_model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=test_texts,
        val_labels=test_labels,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        label_all_tokens=args.label_all_tokens,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        oversample_entities=args.oversample_entities,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
