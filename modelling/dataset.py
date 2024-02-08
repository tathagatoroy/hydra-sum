import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer  # Use a compatible tokenizer if needed
from transformers import RobertaTokenizer, RobertaModel,RobertaForSequenceClassification
import pickle
 
# model = RobertaModel.from_pretrained('roberta-large')
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)


class TextClassificationDataset(Dataset):
    def __init__(self, data_path = "train_head_0.pkl", max_len = 512, vocab_size = 50265):
        """
        Initializes the TextClassificationDataset class.

        Args:
            data_path (str): Path to the text data file.
            label_path (str): Path to the label data file.
            tokenizer (transformers.AutoTokenizer): Transformers tokenizer for text processing.
            max_len (int): Maximum token length for sequences.
            vocab_size (int, optional): Vocabulary size. Defaults to None (use tokenizer's vocabulary).
        """
        self.data_path = data_path
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.max_len = max_len
        self.vocab_size = vocab_size or tokenizer.model_max_len
        self.data_dict = pickle.load(open(data_path, "rb"))
        self.index_to_keys = list(self.data_dict.keys())



    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        #read the pickle file
        return len(self.data_dict.keys())
        

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input tensors for the model:
                - input_ids: Token IDs for the text sequence.
                - attention_mask: Attention mask for padded tokens.
                - labels: Corresponding label tensor.
        """
        text = self.data_dict[self.index_to_keys[idx]]['article']
        label = self.data_dict[self.index_to_keys[idx]]['best_rouge'] - 1

        # Tokenize and encode the text
        encoded_inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'  # Convert to PyTorch tensors
        )

        # Return data dictionary
        return {
            'input_ids': encoded_inputs['input_ids'].squeeze(0),
            'attention_mask': encoded_inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label),
            'index' : self.index_to_keys[idx]
        }

#for testing 
if __name__=="__main__":
    dataset = TextClassificationDataset()
    example = dataset[0]
    print(example['input_ids'].shape)
    print(example['attention_mask'].shape)
    print(example['labels'])