
import os
import ollama

DIRPATH = os.path.dirname(os.path.realpath(__file__))

class TextGenerator:
    @staticmethod
    def generateText(prompt: str, model_name="brooqs/mistral-turkish-v2") -> str:
        response = ollama.generate(model=model_name, prompt=prompt)
        return response["response"]

    @staticmethod
    def generateResponse(prompt, model_name="brooqs/mistral-turkish-v2"):
        try:
            response = ollama.chat(model=model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ])
            return response['message']['content']
        except AttributeError as e:
            return {"error": f"Error in calling ollama.chat: {str(e)}"}
    
    @staticmethod
    def createDummyText(num_words):
        file_path = DIRPATH + 'data/dummyText.txt'
        with open(file_path, 'r') as file:
            text = file.read()
        
        words = text.split()
        result = []
        
        while len(result) < num_words:
            result.extend(words[:num_words - len(result)])
        
        return ' '.join(result[:num_words])


