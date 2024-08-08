import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
from Abyss import Abyss
from ProcessMedia import PdfProcessor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForSeq2SeqLM, BlipProcessor, BlipForConditionalGeneration, MBartForConditionalGeneration, M2M100Tokenizer, M2M100ForConditionalGeneration, pipeline
import torch
import requests
from PIL import Image
class ProcessWithAI:
    @staticmethod
    def NewsAnalysis(modelName="MesutAktas/TurkishNewsAnalysis", text=None, outputFile=None, inputFile=None, labelListStatus=True, labelList=None):

        
        if text is None and inputFile is None:
            raise ValueError("Either 'text' or 'inputFile' must be provided.")
        if text is not None and inputFile is not None:
            raise ValueError("Both 'text' and 'inputFile' cannot be provided simultaneously.")
        if not isinstance(labelListStatus, bool):
            raise ValueError("'labelListStatus' must be a boolean value.")
        if labelList is not None and not isinstance(labelList, dict):
            raise ValueError("'labelList' must be a dictionary.")

        textData = None
        if text is not None:
            textData = text
        elif inputFile is not None:
            if inputFile.lower().endswith('.pdf'):
                try:
                    textData = PDFProcessor.pdfToText(inputFile)
                except Exception as e:
                    raise Exception(f"An error occurred while converting the PDF file '{inputFile}' to text: {e}")
            else:
                try:
                    with open(inputFile, 'r', encoding='utf-8') as f:
                        textData = f.read()
                except FileNotFoundError:
                    raise FileNotFoundError(f"The file '{inputFile}' was not found.")
                except Exception as e:
                    raise Exception(f"An error occurred while reading the file '{inputFile}': {e}")

        if modelName.endswith(".aby"):
            try:
                model, tokenizer, modelDir = Abyss.extractForModel(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while extracting the model '{modelName}': {e}")
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(modelName)
                model = AutoModelForSequenceClassification.from_pretrained(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while loading the model '{modelName}': {e}")
        

        inputs = tokenizer(textData, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val for key, val in inputs.items()}
        model.eval()
        
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = torch.argmax(logits, dim=1).item()
        except Exception as e:
            raise Exception(f"An error occurred during model inference: {e}")

        if modelName.endswith(".aby"):
            try:
                Abyss.removeTemp(modelDir)
            except Exception as e:
                raise Exception(f"An error occurred while removing the temporary directory '{modelDir}': {e}")
        

        if labelListStatus:
            if labelList is None:
                labelList = {
                    "0": "art", 
                    "1": "economy", 
                    "2": "health", 
                    "3": "life", 
                    "4": "magazine",
                    "5": "politics", 
                    "6": "health", 
                    "7": "technology", 
                }
            try:
                predicted_class = labelList[str(predicted_class_idx)]
            except KeyError:
                raise KeyError(f"The predicted class index '{predicted_class_idx}' is not found in the label list.")
        else:
            predicted_class = predicted_class_idx

        if outputFile is not None:
            try:
                with open(outputFile, 'w', encoding='utf-8') as output_file:
                    output_file.write(str(predicted_class))
                    print("Successfully written to file")
            except Exception as e:
                raise Exception(f"An error occurred while writing to the file '{outputFile}': {e}")
        else:
            return predicted_class
    @staticmethod
    def OffensiveLanguageAnalysis(modelName="MesutAktas/TurkishOffensiveLanguageAnalysis", text=None, outputFile=None, inputFile=None, labelListStatus=True, labelList=None):

        
        if text is None and inputFile is None:
            raise ValueError("Either 'text' or 'inputFile' must be provided.")
        if text is not None and inputFile is not None:
            raise ValueError("Both 'text' and 'inputFile' cannot be provided simultaneously.")
        if not isinstance(labelListStatus, bool):
            raise ValueError("'labelListStatus' must be a boolean value.")
        if labelList is not None and not isinstance(labelList, dict):
            raise ValueError("'labelList' must be a dictionary.")

        textData = None
        if text is not None:
            textData = text
        elif inputFile is not None:
            if inputFile.lower().endswith('.pdf'):
                try:
                    textData = PDFProcessor.pdfToText(inputFile)
                except Exception as e:
                    raise Exception(f"An error occurred while converting the PDF file '{inputFile}' to text: {e}")
            else:
                try:
                    with open(inputFile, 'r', encoding='utf-8') as f:
                        textData = f.read()
                except FileNotFoundError:
                    raise FileNotFoundError(f"The file '{inputFile}' was not found.")
                except Exception as e:
                    raise Exception(f"An error occurred while reading the file '{inputFile}': {e}")

        if modelName.endswith(".aby"):
            try:
                model, tokenizer, modelDir = Abyss.extractForModel(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while extracting the model '{modelName}': {e}")
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(modelName)
                model = AutoModelForSequenceClassification.from_pretrained(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while loading the model '{modelName}': {e}")
        
        inputs = tokenizer(textData, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val for key, val in inputs.items()}
        model.eval()
        
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = torch.argmax(logits, dim=1).item()
        except Exception as e:
            raise Exception(f"An error occurred during model inference: {e}")

        if modelName.endswith(".aby"):
            try:
                Abyss.removeTemp(modelDir)
            except Exception as e:
                raise Exception(f"An error occurred while removing the temporary directory '{modelDir}': {e}")
        

        if labelListStatus:
            if labelList is None:
                labelList = {
                    "0": "normal", 
                    "1": "offensive"
                }
            try:
                predicted_class = labelList[str(predicted_class_idx)]
            except KeyError:
                raise KeyError(f"The predicted class index '{predicted_class_idx}' is not found in the label list.")
        else:
            predicted_class = predicted_class_idx

        if outputFile is not None:
            try:
                with open(outputFile, 'w', encoding='utf-8') as output_file:
                    output_file.write(str(predicted_class))
                    print("Successfully written to file")
            except Exception as e:
                raise Exception(f"An error occurred while writing to the file '{outputFile}': {e}")
        else:
            return predicted_class

    @staticmethod
    def DetailedEmotionAnalysis(modelName="MesutAktas/TurkishDetailedEmotionAnalysis", text=None, outputFile=None, inputFile=None, labelListStatus=True, labelList=None):

        
        if text is None and inputFile is None:
            raise ValueError("Either 'text' or 'inputFile' must be provided.")
        if text is not None and inputFile is not None:
            raise ValueError("Both 'text' and 'inputFile' cannot be provided simultaneously.")
        if not isinstance(labelListStatus, bool):
            raise ValueError("'labelListStatus' must be a boolean value.")
        if labelList is not None and not isinstance(labelList, dict):
            raise ValueError("'labelList' must be a dictionary.")

        textData = None
        if text is not None:
            textData = text
        elif inputFile is not None:
            if inputFile.lower().endswith('.pdf'):
                try:
                    textData = PDFProcessor.pdfToText(inputFile)
                except Exception as e:
                    raise Exception(f"An error occurred while converting the PDF file '{inputFile}' to text: {e}")
            else:
                try:
                    with open(inputFile, 'r', encoding='utf-8') as f:
                        textData = f.read()
                except FileNotFoundError:
                    raise FileNotFoundError(f"The file '{inputFile}' was not found.")
                except Exception as e:
                    raise Exception(f"An error occurred while reading the file '{inputFile}': {e}")

        if modelName.endswith(".aby"):
            try:
                model, tokenizer, modelDir = Abyss.extractForModel(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while extracting the model '{modelName}': {e}")
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(modelName)
                model = AutoModelForSequenceClassification.from_pretrained(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while loading the model '{modelName}': {e}")
        

        inputs = tokenizer(textData, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val for key, val in inputs.items()}
        model.eval()
        
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = torch.argmax(logits, dim=1).item()
        except Exception as e:
            raise Exception(f"An error occurred during model inference: {e}")

        if modelName.endswith(".aby"):
            try:
                Abyss.removeTemp(modelDir)
            except Exception as e:
                raise Exception(f"An error occurred while removing the temporary directory '{modelDir}': {e}")
        

        if labelListStatus:
            if labelList is None:
                labelList = {
                    "0": "anger", 
                    "1": "disgust", 
                    "2": "fear", 
                    "3": "joy", 
                    "4": "neutral",
                    "5": "sadness", 
                    "6": "surprise" 
                }
            try:
                predicted_class = labelList[str(predicted_class_idx)]
            except KeyError:
                raise KeyError(f"The predicted class index '{predicted_class_idx}' is not found in the label list.")
        else:
            predicted_class = predicted_class_idx

        if outputFile is not None:
            try:
                with open(outputFile, 'w', encoding='utf-8') as output_file:
                    output_file.write(str(predicted_class))
                    print("Successfully written to file")
            except Exception as e:
                raise Exception(f"An error occurred while writing to the file '{outputFile}': {e}")
        else:
            return predicted_class

    @staticmethod
    def SimpleEmotionAnalysis(modelName="MesutAktas/TurkishSimpleEmotionAnalysis", text=None, outputFile=None, inputFile=None, labelListStatus=True, labelList=None):
        if text is None and inputFile is None:
            raise ValueError("Either 'text' or 'inputFile' must be provided.")
        if text is not None and inputFile is not None:
            raise ValueError("Both 'text' and 'inputFile' cannot be provided simultaneously.")
        if not isinstance(labelListStatus, bool):
            raise ValueError("'labelListStatus' must be a boolean value.")
        if labelList is not None and not isinstance(labelList, dict):
            raise ValueError("'labelList' must be a dictionary.")

        textData = None
        if text is not None:
            textData = text
        elif inputFile is not None:
            if inputFile.lower().endswith('.pdf'):
                try:
                    textData = PDFProcessor.pdfToText(inputFile)
                except Exception as e:
                    raise Exception(f"An error occurred while converting the PDF file '{inputFile}' to text: {e}")
            else:
                try:
                    with open(inputFile, 'r', encoding='utf-8') as f:
                        textData = f.read()
                except FileNotFoundError:
                    raise FileNotFoundError(f"The file '{inputFile}' was not found.")
                except Exception as e:
                    raise Exception(f"An error occurred while reading the file '{inputFile}': {e}")

        if modelName.endswith(".aby"):
            try:
                model, tokenizer, modelDir = Abyss.extractForModel(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while extracting the model '{modelName}': {e}")
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(modelName)
                model = AutoModelForSequenceClassification.from_pretrained(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while loading the model '{modelName}': {e}")
        

        inputs = tokenizer(textData, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val for key, val in inputs.items()}
        model.eval()
        
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = torch.argmax(logits, dim=1).item()
        except Exception as e:
            raise Exception(f"An error occurred during model inference: {e}")

        if modelName.endswith(".aby"):
            try:
                Abyss.removeTemp(modelDir)
            except Exception as e:
                raise Exception(f"An error occurred while removing the temporary directory '{modelDir}': {e}")
        
        if labelListStatus:
            if labelList is None:
                labelList = {
                    "0": "negative", 
                    "1": "neutral", 
                    "2": "positive"
                }
            try:
                predicted_class = labelList[str(predicted_class_idx)]
            except KeyError:
                raise KeyError(f"The predicted class index '{predicted_class_idx}' is not found in the label list.")
        else:
            predicted_class = predicted_class_idx

        if outputFile is not None:
            try:
                with open(outputFile, 'w', encoding='utf-8') as output_file:
                    output_file.write(str(predicted_class))
                    print("Successfully written to file")
            except Exception as e:
                raise Exception(f"An error occurred while writing to the file '{outputFile}': {e}")
        else:
            return predicted_class

    @staticmethod
    def BullyingAnalysis(modelNam="MesutAktas/TurkishBullyingAnalysis", text=None, outputFile=None, inputFile=None, labelListStatus=True, labelList=None):

        
        if text is None and inputFile is None:
            raise ValueError("Either 'text' or 'inputFile' must be provided.")
        if text is not None and inputFile is not None:
            raise ValueError("Both 'text' and 'inputFile' cannot be provided simultaneously.")
        if not isinstance(labelListStatus, bool):
            raise ValueError("'labelListStatus' must be a boolean value.")
        if labelList is not None and not isinstance(labelList, dict):
            raise ValueError("'labelList' must be a dictionary.")

        textData = None
        if text is not None:
            textData = text
        elif inputFile is not None:
            if inputFile.lower().endswith('.pdf'):
                try:
                    textData = PDFProcessor.pdfToText(inputFile)
                except Exception as e:
                    raise Exception(f"An error occurred while converting the PDF file '{inputFile}' to text: {e}")
            else:
                try:
                    with open(inputFile, 'r', encoding='utf-8') as f:
                        textData = f.read()
                except FileNotFoundError:
                    raise FileNotFoundError(f"The file '{inputFile}' was not found.")
                except Exception as e:
                    raise Exception(f"An error occurred while reading the file '{inputFile}': {e}")

        if modelName.endswith(".aby"):
            try:
                model, tokenizer, modelDir = Abyss.extractForModel(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while extracting the model '{modelName}': {e}")
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(modelName)
                model = AutoModelForSequenceClassification.from_pretrained(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while loading the model '{modelName}': {e}")
        

        inputs = tokenizer(textData, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val for key, val in inputs.items()}
        model.eval()
        
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = torch.argmax(logits, dim=1).item()
        except Exception as e:
            raise Exception(f"An error occurred during model inference: {e}")

        if modelName.endswith(".aby"):
            try:
                Abyss.removeTemp(modelDir)
            except Exception as e:
                raise Exception(f"An error occurred while removing the temporary directory '{modelDir}': {e}")
        

        if labelListStatus:
            if labelList is None:
                labelList = {
                    "0": "neutral", 
                    "1": "provocative", 
                    "2": "racism", 
                    "3": "sexism"
                }
            try:
                predicted_class = labelList[str(predicted_class_idx)]
            except KeyError:
                raise KeyError(f"The predicted class index '{predicted_class_idx}' is not found in the label list.")
        else:
            predicted_class = predicted_class_idx

        if outputFile is not None:
            try:
                with open(outputFile, 'w', encoding='utf-8') as output_file:
                    output_file.write(str(predicted_class))
                    print("Successfully written to file")
            except Exception as e:
                raise Exception(f"An error occurred while writing to the file '{outputFile}': {e}")
        else:
            return predicted_class


    @staticmethod
    def TranslateToTurkish(modelName="Helsinki-NLP/opus-tatoeba-en-tr", text=None, outputFile=None, inputFile=None, labelListStatus=False, labelList=None):

        
        if text is None and inputFile is None:
            raise ValueError("Either 'text' or 'inputFile' must be provided.")
        if text is not None and inputFile is not None:
            raise ValueError("Both 'text' and 'inputFile' cannot be provided simultaneously.")
        if not isinstance(labelListStatus, bool):
            raise ValueError("'labelListStatus' must be a boolean value.")
        if labelList is not None and not isinstance(labelList, dict):
            raise ValueError("'labelList' must be a dictionary.")

        textData = None
        if text is not None:
            textData = text
        elif inputFile is not None:
            if inputFile.lower().endswith('.pdf'):
                try:
                    textData = PDFProcessor.pdfToText(inputFile)
                except Exception as e:
                    raise Exception(f"An error occurred while converting the PDF file '{inputFile}' to text: {e}")
            else:
                try:
                    with open(inputFile, 'r', encoding='utf-8') as f:
                        textData = f.read()
                except FileNotFoundError:
                    raise FileNotFoundError(f"The file '{inputFile}' was not found.")
                except Exception as e:
                    raise Exception(f"An error occurred while reading the file '{inputFile}': {e}")

        if modelName.endswith(".aby"):
            try:
                model, tokenizer, modelDir = Abyss.extractForModel(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while extracting the model '{modelName}': {e}")
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(modelName)
                model = AutoModelForSeq2SeqLM.from_pretrained(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while loading the model '{modelName}': {e}")
        

        inputs = tokenizer(textData, return_tensors="pt", truncation=True, padding=True)
        translated_tokens = model.generate(inputs["input_ids"])
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        if modelName.endswith(".aby"):
            try:
                Abyss.removeTemp(modelDir)
            except Exception as e:
                raise Exception(f"An error occurred while removing the temporary directory '{modelDir}': {e}")
        
        if outputFile is not None:
            try:
                with open(outputFile, 'w', encoding='utf-8') as output_file:
                    output_file.write(translated_text[0])
                    print("Successfully written to file")
            except Exception as e:
                raise Exception(f"An error occurred while writing to the file '{outputFile}': {e}")
        else:
            return translated_text[0]

    @staticmethod
    def TranslateToEnglish(modelName="Helsinki-NLP/opus-mt-tr-en", text=None, outputFile=None, inputFile=None, labelListStatus=True, labelList=None):

        
        if text is None and inputFile is None:
            raise ValueError("Either 'text' or 'inputFile' must be provided.")
        if text is not None and inputFile is not None:
            raise ValueError("Both 'text' and 'inputFile' cannot be provided simultaneously.")
        if not isinstance(labelListStatus, bool):
            raise ValueError("'labelListStatus' must be a boolean value.")
        if labelList is not None and not isinstance(labelList, dict):
            raise ValueError("'labelList' must be a dictionary.")

        textData = None
        if text is not None:
            textData = text
        elif inputFile is not None:
            if inputFile.lower().endswith('.pdf'):
                try:
                    textData = PDFProcessor.pdfToText(inputFile)
                except Exception as e:
                    raise Exception(f"An error occurred while converting the PDF file '{inputFile}' to text: {e}")
            else:
                try:
                    with open(inputFile, 'r', encoding='utf-8') as f:
                        textData = f.read()
                except FileNotFoundError:
                    raise FileNotFoundError(f"The file '{inputFile}' was not found.")
                except Exception as e:
                    raise Exception(f"An error occurred while reading the file '{inputFile}': {e}")

        if modelName.endswith(".aby"):
            try:
                model, tokenizer, modelDir = Abyss.extractForModel(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while extracting the model '{modelName}': {e}")
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(modelName)
                model = AutoModelForSeq2SeqLM.from_pretrained(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while loading the model '{modelName}': {e}")
        

        inputs = tokenizer(textData, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v for k, v in inputs.items()}  
        translated_tokens = model.generate(inputs["input_ids"])
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        if modelName.endswith(".aby"):
            try:
                Abyss.removeTemp(modelDir)
            except Exception as e:
                raise Exception(f"An error occurred while removing the temporary directory '{modelDir}': {e}")
        
        if outputFile is not None:
            try:
                with open(outputFile, 'w', encoding='utf-8') as output_file:
                    output_file.write(translated_text[0])
                    print("Successfully written to file")
            except Exception as e:
                raise Exception(f"An error occurred while writing to the file '{outputFile}': {e}")
        else:
            return translated_text[0]

    @staticmethod
    def ImageToText(self, img_path):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        image = Image.open(img_path)

        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        translated_text = self.TranslateToTurkish(caption)

        return translated_text
    @staticmethod
    def NERAnalysis(modelName= "MesutAktas/TurkishNERModel", text=None, outputFile=None, inputFile=None, labelListStatus=True, labelList=None):
    
            
            if text is None and inputFile is None:
                raise ValueError("Either 'text' or 'inputFile' must be provided.")
            if text is not None and inputFile is not None:
                raise ValueError("Both 'text' and 'inputFile' cannot be provided simultaneously.")
            if not isinstance(labelListStatus, bool):
                raise ValueError("'labelListStatus' must be a boolean value.")
            if labelList is not None and not isinstance(labelList, dict):
                raise ValueError("'labelList' must be a dictionary.")
            
            textData = None
            if text is not None:
                textData = text
            elif inputFile is not None:
                if inputFile.lower().endswith('.pdf'):
                    try:
                        textData = pdfToText(inputFile)
                    except Exception as e:
                        raise Exception(f"An error occurred while converting the PDF file '{inputFile}' to text: {e}")
                else:
                    try:
                        with open(inputFile, 'r', encoding='utf-8') as f:
                            textData = f.read()
                    except FileNotFoundError:
                        raise FileNotFoundError(f"The file '{inputFile}' was not found.")
                    except Exception as e:
                        raise Exception(f"An error occurred while reading the file '{inputFile}': {e}")

            if modelName.endswith(".aby"):
                try:
                    ner_pipeline, modelDir = Abyss.extractForNerModel(modelName)
                except Exception as e:
                    raise Exception(f"An error occurred while extracting the model '{modelName}': {e}")
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(modelName)
                    model = AutoModelForTokenClassification.from_pretrained(modelName)
                    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
                except Exception as e:
                    raise Exception(f"An error occurred while loading the model '{modelName}': {e}")
            
            try:
                ner_results = ner_pipeline(textData)
                entities = [(entity['word'], entity['entity']) for entity in ner_results]
            except Exception as e:
                raise Exception(f"An error occurred during model inference: {e}")

            if modelName.endswith(".aby"):
                try:
                    Abyss.removeTemp(modelDir)
                except Exception as e:
                    raise Exception(f"An error occurred while removing the temporary directory '{modelDir}': {e}")
            
    
            if labelListStatus:
                if labelList is None:
                    labelList = {
                        "LABEL_0": "O",
                        "LABEL_1": "B-PERSON",
                        "LABEL_2": "I-PERSON",
                        "LABEL_3": "B-LOCATION",
                        "LABEL_4": "I-LOCATION",
                        "LABEL_5": "B-ORGANIZATION",
                        "LABEL_6": "I-ORGANIZATION",
                        "LABEL_7": "B-MISC",
                        "LABEL_8": "I-MISC"
                    }
                try:
                    entities = [(word, labelList[entity]) for word, entity in entities]
                except KeyError as e:
                    raise KeyError(f"The predicted entity '{e}' is not found in the label list.")

            if outputFile is not None:
                try:
                    with open(outputFile, 'w', encoding='utf-8') as output_file:
                        for word, entity in entities:
                            output_file.write(f"{word}: {entity}\n")
                        print("Successfully written to file")
                except Exception as e:
                    raise Exception(f"An error occurred while writing to the file '{outputFile}': {e}")
            else:
                return entities
    @staticmethod
    def EntityBasedSentimentAnalysis(nerModelName='MesutAktas/TurkishNERModel', sentimentModelName='MesutAktas/TurkishDetailedEmotionAnalysis', text=None, inputFile=None, outputFile=None, labelListStatus=True, labelList=None, sentimentLabelList=None):

        if text is None and inputFile is None:
            raise ValueError("Either 'text' or 'inputFile' must be provided.")
        if text is not None and inputFile is not None:
            raise ValueError("Both 'text' and 'inputFile' cannot be provided simultaneously.")
        if not isinstance(labelListStatus, bool):
            raise ValueError("'labelListStatus' must be a boolean value.")
        if labelList is not None and not isinstance(labelList, dict):
            raise ValueError("'labelList' must be a dictionary.")
        sentimentLabelList = {
            "0": 'anger',
            "1": 'disgust',
            "2": 'fear',
            "3": 'joy',
            "4": 'neutral',
            "5": 'sadness',
            "6": 'surprise'
        }
        textData = None
        if text is not None:
            textData = text
        elif inputFile is not None:
            if inputFile.lower().endswith('.pdf'):
                try:
                    textData = pdfToText(inputFile)
                except Exception as e:
                    raise Exception(f"An error occurred while converting the PDF file '{inputFile}' to text: {e}")
            else:
                try:
                    with open(inputFile, 'r', encoding='utf-8') as f:
                        textData = f.read()
                except FileNotFoundError:
                    raise FileNotFoundError(f"The file '{inputFile}' was not found.")
                except Exception as e:
                    raise Exception(f"An error occurred while reading the file '{inputFile}': {e}")

        if nerModelName.endswith(".aby"):
            try:
                nerModel, nerModelDir = Abyss.extractForNerModel(nerModelName)
            except Exception as e:
                raise Exception(f"An error occurred while extracting the NER model '{nerModelName}': {e}")
        else:
            try:
                nerModel = pipeline('ner', model=nerModelName, tokenizer=nerModelName, device=0 if torch.cuda.is_available() else -1)
            except Exception as e:
                raise Exception(f"An error occurred while loading the NER model '{nerModelName}': {e}")

        if sentimentModelName.endswith(".aby"):
            try:
                sentimentModel, sentimentTokenizer, sentimentModelDir = Abyss.extractForModel(sentimentModelName)
                sentimentModel
            except Exception as e:
                raise Exception(f"An error occurred while extracting the Sentiment model '{sentimentModelName}': {e}")
        else:
            try:
                sentimentModel = AutoModelForSequenceClassification.from_pretrained(sentimentModelName)
                sentimentTokenizer = AutoTokenizer.from_pretrained(sentimentModelName)
                sentimentModel
            except Exception as e:
                raise Exception(f"An error occurred while loading the Sentiment model '{sentimentModelName}': {e}")

        ner_results = nerModel(textData)
        entities = []
        current_entity = ""
        entity_labels = ["LABEL_1", "LABEL_3", "LABEL_5"]
        inside_labels = ["LABEL_2", "LABEL_4", "LABEL_6"]

        for result in ner_results:
            word = result['word']
            label = result['entity']

            if any(entity_label in label for entity_label in entity_labels):
                if current_entity:
                    entities.append(current_entity)
                current_entity = word
            elif any(inside_label in label for inside_label in inside_labels) and current_entity:
                current_entity += " " + word
            elif 'LABEL_0' in label and current_entity:
                current_entity += " " + word
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = ""
        if current_entity:
            entities.append(current_entity)

        results = []
        for entity in entities:
            tokens = sentimentTokenizer(entity, padding=True, truncation=True, return_tensors="pt")
            tokens = {k: v for k, v in tokens.items()}

            with torch.no_grad():
                outputs = sentimentModel(**tokens)
                sentiment_label_index = outputs.logits.argmax(dim=1).item()
                sentiment_label = list(sentimentLabelList.values())[sentiment_label_index] if sentimentLabelList else str(sentiment_label_index)
                result = {
                    "Entity": entity,
                    "Sentiment": sentiment_label
                }
                results.append(result)

        if outputFile is not None:
            try:
                with open(outputFile, 'w', encoding='utf-8') as output_file:
                    for result in results:
                        output_file.write(result + "\n")
                    print("Successfully written to file")
            except Exception as e:
                raise Exception(f"An error occurred while writing to the file '{outputFile}': {e}")
        else:
            return results

        if nerModelName.endswith(".aby"):
            try:
                Abyss.removeTemp(nerModelDir)
            except Exception as e:
                raise Exception(f"An error occurred while removing the temporary directory '{nerModelDir}': {e}")

        if sentimentModelName.endswith(".aby"):
            try:
                Abyss.removeTemp(sentimentModelDir)
            except Exception as e:
                raise Exception(f"An error occurred while removing the temporary directory '{sentimentModelDir}': {e}")
    @staticmethod
    def CustomAnalysis(modelName=None, text=None, outputFile=None, inputFile=None, labelListStatus=False, labelList=None):

        if text is None and inputFile is None:
            raise ValueError("Either 'text' or 'inputFile' must be provided.")
        if text is not None and inputFile is not None:
            raise ValueError("Both 'text' and 'inputFile' cannot be provided simultaneously.")
        if not isinstance(labelListStatus, bool):
            raise ValueError("'labelListStatus' must be a boolean value.")
        if labelList is not None and not isinstance(labelList, dict):
            raise ValueError("'labelList' must be a dictionary.")

        textData = None
        if text is not None:
            textData = text
        elif inputFile is not None:
            if inputFile.lower().endswith('.pdf'):
                try:
                    textData = PDFProcessor.pdfToText(inputFile)
                except Exception as e:
                    raise Exception(f"An error occurred while converting the PDF file '{inputFile}' to text: {e}")
            else:
                try:
                    with open(inputFile, 'r', encoding='utf-8') as f:
                        textData = f.read()
                except FileNotFoundError:
                    raise FileNotFoundError(f"The file '{inputFile}' was not found.")
                except Exception as e:
                    raise Exception(f"An error occurred while reading the file '{inputFile}': {e}")

        if modelName.endswith(".aby"):
            try:
                model, tokenizer, modelDir = Abyss.extractForModel(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while extracting the model '{modelName}': {e}")
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(modelName)
                model = AutoModelForSequenceClassification.from_pretrained(modelName)
            except Exception as e:
                raise Exception(f"An error occurred while loading the model '{modelName}': {e}")
        

        inputs = tokenizer(textData, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: val for key, val in inputs.items()}
        model.eval()
        
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = torch.argmax(logits, dim=1).item()
        except Exception as e:
            raise Exception(f"An error occurred during model inference: {e}")

        if modelName.endswith(".aby"):
            try:
                Abyss.removeTemp(modelDir)
            except Exception as e:
                raise Exception(f"An error occurred while removing the temporary directory '{modelDir}': {e}")
        

        if labelListStatus:
            if labelList is None:
                labelList = {}
            try:
                predicted_class = labelList[str(predicted_class_idx)]
            except KeyError:
                raise KeyError(f"The predicted class index '{predicted_class_idx}' is not found in the label list.")
        else:
            predicted_class = predicted_class_idx

        if outputFile is not None:
            try:
                with open(outputFile, 'w', encoding='utf-8') as output_file:
                    output_file.write(str(predicted_class))
                    print("Successfully written to file")
            except Exception as e:
                raise Exception(f"An error occurred while writing to the file '{outputFile}': {e}")
        else:
            return predicted_class
