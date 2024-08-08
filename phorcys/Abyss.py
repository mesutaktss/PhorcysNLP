import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
import struct
import tempfile
import shutil
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline

class Abyss:
    _temp_dir = None

    @staticmethod
    def compressFolder(folder_path, aby_path):
        files = []
        
        # Read all files in the folder
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                with open(file_path, 'rb') as f:
                    files.append((os.path.relpath(file_path, folder_path), f.read()))

        # Create .aby file and write header
        with open(aby_path, 'wb') as aby_file:
            # Write the number of files
            aby_file.write(struct.pack('I', len(files)))

            for file_name, file_data in files:
                # Write the file name length and file name
                aby_file.write(struct.pack('I', len(file_name)))
                aby_file.write(file_name.encode('utf-8'))
                
                # Write the data length and data
                aby_file.write(struct.pack('I', len(file_data)))
                aby_file.write(file_data)

    @staticmethod
    def extractAby(aby_path, extract_to=None):
        # Create a directory with the same name as the .aby file if no extract_to is provided
        if extract_to is None:
            extract_to = aby_path[:-4]  # Remove the .aby extension for the folder name
        
        os.makedirs(extract_to, exist_ok=True)
        
        # Open .aby file and read header
        with open(aby_path, 'rb') as aby_file:
            # Read the number of files
            num_files = struct.unpack('I', aby_file.read(4))[0]
            
            for _ in range(num_files):
                # Read the file name length and file name
                name_len = struct.unpack('I', aby_file.read(4))[0]
                file_name = aby_file.read(name_len).decode('utf-8')
                
                # Read the data length and data
                data_len = struct.unpack('I', aby_file.read(4))[0]
                file_data = aby_file.read(data_len)
                
                # Write the data to the appropriate file
                file_path = os.path.join(extract_to, file_name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(file_data)

    @staticmethod
    def extractForModel(aby_path):
        if Abyss._temp_dir is None:
            # Create a temporary directory
            Abyss._temp_dir = tempfile.mkdtemp(prefix='.model_', dir=os.getcwd())
            Abyss.extractAby(aby_path, extract_to=Abyss._temp_dir)

        model = AutoModelForSequenceClassification.from_pretrained(Abyss._temp_dir)
        tokenizer = AutoTokenizer.from_pretrained(Abyss._temp_dir)
        return model, tokenizer, Abyss._temp_dir

    @staticmethod
    def extractForNerModel(aby_path):
        if Abyss._temp_dir is None:
            # Create a temporary directory
            Abyss._temp_dir = tempfile.mkdtemp(prefix='.model_', dir=os.getcwd())
            Abyss.extractAby(aby_path, extract_to=Abyss._temp_dir)

        pipe = pipeline('ner', model=Abyss._temp_dir, tokenizer=Abyss._temp_dir)
        return pipe, Abyss._temp_dir

    @staticmethod
    def modelCompress(model_name_or_path, aby_path):
        aby_path = aby_path + ".aby"
        # Create a temporary directory to store the model files
        temp_dir = 'temp_model_dir'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the tokenizer and model to the temporary directory
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
        
        tokenizer.save_pretrained(temp_dir)
        model.save_pretrained(temp_dir)
        
        # Compress the temporary directory into a .aby file
        files = []
        for root, _, filenames in os.walk(temp_dir):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                with open(file_path, 'rb') as f:
                    files.append((os.path.relpath(file_path, temp_dir), f.read()))

        with open(aby_path, 'wb') as aby_file:
            aby_file.write(struct.pack('I', len(files)))
            for file_name, file_data in files:
                aby_file.write(struct.pack('I', len(file_name)))
                aby_file.write(file_name.encode('utf-8'))
                aby_file.write(struct.pack('I', len(file_data)))
                aby_file.write(file_data)

        shutil.rmtree(temp_dir)

    @staticmethod
    def removeTemp(path):
        shutil.rmtree(path)
