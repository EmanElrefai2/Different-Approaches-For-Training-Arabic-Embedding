from ragatouille import RAGTrainer
from datasets import load_dataset

def main():
    trainer = RAGTrainer(model_name="ArabicColBERT_WideBot", pretrained_model_name="UBC-NLP/MARBERT", language_code="ar")

    sample_size = 100000
    ds = load_dataset('unicamp-dl/mmarco', 'arabic', split="train", trust_remote_code=True, streaming=True)
    sds = ds.shuffle(seed=42, buffer_size=10_000)
    dataset = sds.take(sample_size)
    print(dataset)

    # Initialize lists to hold queries, positive passages, and negative passages
    queries = []
    positive_passages = []
    negative_passages = []

    # Iterate through the dataset and extract the required fields
    for sample in dataset:
        queries.append(sample['query'])
        positive_passages.append(sample['positive'])
        negative_passages.append(sample['negative'])

    # Create pairs
    pairs = []

    for query, positive, negative in zip(queries, positive_passages, negative_passages):
        pairs.append((query, positive))
        pairs.append((query, negative))

    # Print pairs to verify
    for pair in pairs[:10]:  # Printing only the first 10 pairs to avoid excessive output
        print(pair)

    trainer.prepare_training_data(raw_data=pairs, data_out_path="./data_100000/", num_new_negatives=10, mine_hard_negatives=True)
    trainer.train(batch_size=32,
                  nbits=4,
                  maxsteps=500000,
                  use_ib_negatives=True,
                  dim=128,
                  learning_rate=5e-6,
                  doc_maxlen=512,
                  use_relu=False,
                  warmup_steps="auto")

if __name__ == "__main__":
    main()
