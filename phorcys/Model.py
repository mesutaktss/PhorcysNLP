import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class ModelGen:
    @staticmethod
    def generateClassificationModel(dataPath, modelName, labelColumn, textColumn, saveDir="generatedModel", batchSize=12, maxLength=60, numEpochs=3, learningRate=1e-5):
        badLines = []

        # Handle bad lines
        def handleBadLines(line, lineNum):
            badLines.append(line)
            return None

        df = pd.read_csv(dataPath, on_bad_lines=handleBadLines, engine='python')
        df = df.applymap(str)
        # Save bad lines
        with open("badLines.txt", "w") as f:
            for line in badLines:
                f.write(line + "\n")

        # Label encoding
        labelEncoder = LabelEncoder()
        df[labelColumn] = labelEncoder.fit_transform(df[labelColumn])

        labelMapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
        with open("labelMapping.txt", "w") as f:
            for label, idx in labelMapping.items():
                f.write(f"{label}: {idx}\n")

        # Split the data
        trainDf, testDf = train_test_split(df, test_size=0.2, random_state=42)
        trainDf, validDf = train_test_split(trainDf, test_size=0.1, random_state=42)

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(modelName)
        model = AutoModelForSequenceClassification.from_pretrained(modelName, num_labels=len(df[labelColumn].unique()))

        # Tokenize the data
        trainEncodings = tokenizer(list(trainDf[textColumn]), truncation=True, padding=True, max_length=maxLength)
        validEncodings = tokenizer(list(validDf[textColumn]), truncation=True, padding=True, max_length=maxLength)
        testEncodings = tokenizer(list(testDf[textColumn]), truncation=True, padding=True, max_length=maxLength)

        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        trainDataset = CustomDataset(trainEncodings, list(trainDf[labelColumn]))
        validDataset = CustomDataset(validEncodings, list(validDf[labelColumn]))
        testDataset = CustomDataset(testEncodings, list(testDf[labelColumn]))

        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
        validLoader = torch.utils.data.DataLoader(validDataset, batch_size=batchSize, shuffle=False)
        testLoader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, shuffle=False)

        # Move model to GPU
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        # Define optimizer
        optimizer = AdamW(model.parameters(), lr=learningRate)

        print("Model training started")

        def evaluateModel(loader):
            model.eval()
            totalLoss = 0
            correct = 0
            total = 0
            allLabels = []
            allPredictions = []

            with torch.no_grad():
                for batch in tqdm(loader, desc="Evaluating", unit="batch"):
                    inputIds = batch['input_ids'].to(device)
                    attentionMask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(inputIds, attention_mask=attentionMask, labels=labels)
                    loss = outputs.loss
                    totalLoss += loss.item()
                    _, predicted = torch.max(outputs.logits, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    allLabels.extend(labels.cpu().numpy())
                    allPredictions.extend(predicted.cpu().numpy())

            avgLoss = totalLoss / len(loader)
            accuracy = correct / total
            return avgLoss, accuracy, allLabels, allPredictions

        train_losses = []
        valid_losses = []
        valid_accuracies = []

        # Training loop
        for epoch in range(numEpochs):
            model.train()
            totalTrainLoss = 0
            for batch in tqdm(trainLoader, desc=f"Epoch {epoch+1}/{numEpochs} - Training", unit="batch"):
                optimizer.zero_grad()
                inputIds = batch['input_ids'].to(device)
                attentionMask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(inputIds, attention_mask=attentionMask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                totalTrainLoss += loss.item()

            avgTrainLoss = totalTrainLoss / len(trainLoader)
            train_losses.append(avgTrainLoss)
            print(f"Epoch {epoch+1} completed. Training Loss: {avgTrainLoss}")

            # Validation loop
            avgValidLoss, validAccuracy, _, _ = evaluateModel(validLoader)
            valid_losses.append(avgValidLoss)
            valid_accuracies.append(validAccuracy)
            print(f"Epoch {epoch+1} completed. Validation Loss: {avgValidLoss}, Validation Accuracy: {validAccuracy}")

        # Evaluate on test set
        testLoss, testAccuracy, testLabels, testPredictions = evaluateModel(testLoader)
        print(f"Test Loss: {testLoss}, Test Accuracy: {testAccuracy}")

        # Classification report and confusion matrix
        report = classification_report(testLabels, testPredictions, target_names=labelEncoder.classes_)
        confMatrix = confusion_matrix(testLabels, testPredictions)

        # Save results
        with open("evaluationResults.txt", "w") as f:
            f.write(f"Test Loss: {testLoss}\n")
            f.write(f"Test Accuracy: {testAccuracy}\n")
            f.write("\nClassification Report:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(confMatrix))

        # Make tensors contiguous
        def makeContiguous(model):
            for param in model.parameters():
                param.data = param.data.contiguous()

        makeContiguous(model)

        # Save model and tokenizer
        model.save_pretrained(saveDir)
        tokenizer.save_pretrained(saveDir)

        print("Model saved")

        # Plot training progress
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(numEpochs), train_losses, label='Training Loss')
        plt.plot(range(numEpochs), valid_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.subplot(1, 2, 2)
        plt.plot(range(numEpochs), valid_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Validation Accuracy')

        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()
