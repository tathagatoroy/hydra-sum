from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from dataset import TextClassificationDataset  
import tqdm
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
import pickle


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    res = {}

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            index = batch['index']
            

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            size = len(index)
            for i in range(size):
                res[index[i]] = preds[i].item() + 1
            correct_preds += torch.sum(preds == labels).item()
            total_preds += len(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_preds / total_preds
    #print(f"Val Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f}")
    data_dict = pickle.load(open("valid_head_0.pkl", "rb"))
    best = []
    worst = []
    beam_1 = []
    beam_2 = []
    beam_3 = []
    beam_4 = []
    beam_5 = []
    beam_6 = []
    preds = []
    for key in res.keys():
        val = res[key]
        scores = data_dict[str(key)]['rouge_scores']
        rouge_1_scores = [(int(k),x['rouge_1_f_score']) for k,x in scores.items()]
    #sort by the rouge-1 score in descending order
        beam_1.append(scores['1']['rouge_1_f_score'])
        beam_2.append(scores['2']['rouge_1_f_score'])
        beam_3.append(scores['3']['rouge_1_f_score'])
        beam_4.append(scores['4']['rouge_1_f_score'])
        beam_5.append(scores['5']['rouge_1_f_score'])
        beam_6.append(scores['6']['rouge_1_f_score'])
        pred_rouge_score = scores[str(val)]['rouge_1_f_score']
        preds.append(pred_rouge_score)
        
        rouge_1_scores = sorted(rouge_1_scores, key=lambda x: x[1], reverse=True)
        best_score = rouge_1_scores[0][1]
        worst_score = rouge_1_scores[-1][1]

        best.append((best_score))
        worst.append((worst_score))
    print("best rouge-1 score",sum(best)/len(best))
    print("worst rouge-1 score",sum(worst)/len(worst))
    print("beam_1",sum(beam_1)/len(beam_1))
    print("beam_2",sum(beam_2)/len(beam_2))
    print("beam_3",sum(beam_3)/len(beam_3))
    print("beam_4",sum(beam_4)/len(beam_4))
    print("beam_5",sum(beam_5)/len(beam_5))
    print("beam_6",sum(beam_6)/len(beam_6))
    print("preds",sum(preds)/len(preds))
    return avg_loss, accuracy, res


if __name__ == "__main__":
    model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=6)
    
    # Load dataset
    train_dataset = TextClassificationDataset(data_path="train_head_0.pkl", max_len=512, vocab_size=50265)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    print("size of train_set",len(train_dataset))
    
    val_dataset = TextClassificationDataset(data_path="valid_head_0.pkl", max_len=512, vocab_size=50265)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print("size of val_set",len(val_dataset))

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, capturable=True)
    criterion = nn.CrossEntropyLoss()
    print(model.config)
    print(model.config.num_labels)

    # Define evaluation function

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    best_accuracy = 0.0
            # Evaluate after each epoch
    val_loss, val_accuracy, res= evaluate(model, val_loader, device)
    print(f"Epoch {0}/{3}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    for epoch in range(3):  # Adjust the number of epochs as needed
        total_loss = 0
        print(f"Epoch {epoch + 1}/{3}")
        for batch in tqdm.tqdm(train_loader):
            model.train()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{5}, Train Loss: {avg_loss:.4f}")

        # Evaluate after each epoch
        val_loss, val_accuracy, res = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{5}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save the best model based on validation accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model.save_pretrained("roberta_large_best_2")

    # Save the final trained model
    model.save_pretrained("roberta_large_final_2")