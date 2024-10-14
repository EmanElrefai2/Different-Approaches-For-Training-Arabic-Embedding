import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import RerankingEvaluator
import pyarabic.araby as araby


class WordEmbeddingTrainer:
    def __init__(self) -> None:
        self.lr = 2e-05
        self.weight_decay = 0.01
        self.data_folder = "dataset/Train"
        self.embed_models = ["intfloat/multilingual-e5-small"] 
        self.guide_model = "aubmindlab/bert-large-arabertv02"
        self.batch_size = 16
        self.epochs = 10
    

    def preprocess_text(self, text):
        corrected_text = araby.normalize_hamza(str(text))
        corrected_text = araby.strip_tashkeel(corrected_text)
        corrected_text = araby.tokenize(corrected_text)
        
        return " ".join(corrected_text)
    
    def remove_model_safetensors(self, directory):
        for filename in os.listdir(directory):
            if filename == "model.safetensors":
                file_path = os.path.join(directory, filename)
                os.remove(file_path)
                print(f"Removed {file_path}")


    def prepare_eval_data(self):
        samples = []
        for test_df in os.listdir("dataset/Test"):
            if test_df == ".ipynb_checkpoints":
                    continue
        
            df = pd.read_csv(f"dataset/Test/{test_df}")
            for col in df.columns:
                if col.startswith("Unnamed"):
                    continue
                queries = df[col].dropna().to_list()
                for q in queries:
                    samples.append(
                        {
                            "query": self.preprocess_text(q),
                            "positive": [self.preprocess_text(col)],
                            "negative": [self.preprocess_text(column) for column in df.columns if column != col]
                        }
                    )
        
        return samples

    

    def run(self, dataset_path: str, embed_model: str, lr: float, weight_decay: float):
        os.makedirs("Results", exist_ok=True)
        os.makedirs(f"Results/GISTEmbed/{dataset_path.split('/')[-1].split('.')[0]}", exist_ok=True)
        os.makedirs(f"Results/GISTEmbed/{dataset_path.split('/')[-1].split('.')[0]}/{embed_model.split('/')[1]}", exist_ok=True)


        model_save_path = os.path.join("Results", "GISTEmbed", dataset_path.split('/')[-1].split('.')[0], embed_model.split('/')[1])
        
        checkpoints_save_path = os.path.join(model_save_path, "Checkpoints")

        # if not os.path.exists(model_save_path):

        word_embedding_model = models.Transformer(embed_model)

        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )

        t_model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda")
        g_model = SentenceTransformer(self.guide_model)

        train_samples = []

        df = pd.read_csv(dataset_path)

        for _, row in df.iterrows():
            inp_example = InputExample(texts=[row["query"], row["positive"], row["negative"]])
            train_samples.append(inp_example)

        # if embed_model.split('/')[1] == "LaBSE":
        #     batch_size = 128
        # else:
        #     batch_size = 128
        
        train_dataloader = DataLoader(train_samples, batch_size=self.batch_size)
        train_loss = losses.GISTEmbedLoss(t_model, g_model)

        warmup_steps = math.ceil(len(train_dataloader) * self.epochs * 0.1)  # 10% of train data for warm-up

        # samples = self.prepare_eval_data()
        # evaluator = RerankingEvaluator(samples=samples, at_k=3, name="lm_evaluator")


        # Train the model
        t_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            # evaluator=evaluator,
            epochs=self.epochs,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            show_progress_bar=True,
            checkpoint_path=checkpoints_save_path,
            checkpoint_save_steps=len(train_dataloader),
            optimizer_params={'lr': lr},
            weight_decay=weight_decay
        )

        del t_model
        torch.cuda.empty_cache()

        self.remove_model_safetensors(model_save_path)

        # else:
        #     print("Model exists")


    
    def train(self):
        for df in os.listdir(self.data_folder):
            if df.endswith(".csv"):
                for embed_model in self.embed_models:
                    self.run(os.path.join(self.data_folder, df), embed_model, self.lr, self.weight_decay)


Trainer = WordEmbeddingTrainer()
Trainer.train()