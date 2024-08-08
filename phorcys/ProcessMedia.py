import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
import shutil
import subprocess
import cv2
import whisper
from pydub import AudioSegment
import pytesseract
from pytesseract import Output
import fitz
from gtts import gTTS

class MediaToText:
    model = None
    modelName = "medium"
    device = "cpu"

    @staticmethod
    def checkFfmpeg():
        if shutil.which("ffmpeg") is None:
            raise EnvironmentError("ffmpeg is not installed. Please install it from https://ffmpeg.org/download.html.")

    @staticmethod
    def convertToWav(inputFile):
        if not os.path.isfile(inputFile):
            raise FileNotFoundError(f"Input file {inputFile} not found.")
        wavFile = "temp.wav"
        command = f"ffmpeg -y -i {inputFile} -vn -acodec pcm_s16le -ar 44100 -ac 2 {wavFile}"
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to convert {inputFile} to WAV format. Error: {result.stderr.decode('utf-8')}")
        return wavFile

    @staticmethod
    def transcribeAudio(inputFile, modelName=None, device=None):
        if modelName:
            MediaToText.modelName = modelName
        if device:
            MediaToText.device = device
        if MediaToText.model is None or MediaToText.modelName != modelName or MediaToText.device != device:
            MediaToText.model = whisper.load_model(MediaToText.modelName, device=MediaToText.device)
        wavFile = MediaToText.convertToWav(inputFile)
        result = MediaToText.model.transcribe(wavFile, language="tr")
        os.remove(wavFile)
        return result["text"]

    @staticmethod
    def formatTime(seconds):
        millisec = int((seconds % 1) * 1000)
        timeStr = f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02},{millisec:03}"
        return timeStr

    @staticmethod
    def createSubtitles(videoPath, outputFile, segmentDuration=5):
        MediaToText.checkFfmpeg()
        if not os.path.isfile(videoPath):
            raise FileNotFoundError(f"Video file {videoPath} not found.")
        video = cv2.VideoCapture(videoPath)
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {videoPath}")
        fps = video.get(cv2.CAP_PROP_FPS)
        frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0 or frameCount == 0:
            raise ValueError(f"Invalid video file: {videoPath}")
        duration = frameCount / fps
        tempDir = "temp_audio_segments"
        os.makedirs(tempDir, exist_ok=True)
        currentTime = 0
        subtitleLines = []
        srtLines = []
        segmentIndex = 1
        while currentTime < duration:
            startTime = currentTime
            endTime = min(startTime + segmentDuration, duration)
            segmentFilename = os.path.join(tempDir, f"segment_{startTime}_{endTime}.wav")
            command = f"ffmpeg -y -i {videoPath} -ss {startTime} -to {endTime} -q:a 0 -map a {segmentFilename}"
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to extract audio segment from {videoPath}. Error: {result.stderr.decode('utf-8')}")
            try:
                text = MediaToText.transcribeAudio(segmentFilename)
            except Exception as e:
                raise RuntimeError(f"Error transcribing audio segment: {str(e)}")
            subtitleLines.append(f"{startTime:.2f} --> {endTime:.2f}\n{text}\n")
            srtLines.append(f"{segmentIndex}\n{MediaToText.formatTime(startTime)} --> {MediaToText.formatTime(endTime)}\n{text}\n")
            currentTime += segmentDuration
            segmentIndex += 1
        txtOutput = outputFile + ".txt"
        srtOutput = outputFile + ".srt"
        with open(txtOutput, 'w', encoding='utf-8') as f:
            for line in subtitleLines:
                f.write(line)
        with open(srtOutput, 'w', encoding='utf-8') as f:
            for line in srtLines:
                f.write(line)
        for file in os.listdir(tempDir):
            os.remove(os.path.join(tempDir, file))
        os.rmdir(tempDir)

    @staticmethod
    def textToSpeech(text):
        tts = gTTS(text=text, lang='tr', slow=False)
        tts.save("turkce_tts.mp3")


class OcrProcessor:
    def __init__(self, outputText=False, saveBoxes=False):
        self.outputText = outputText
        self.saveBoxes = saveBoxes
        self.setupTesseract()

    @staticmethod
    def setupTesseract():
        if os.name == 'posix':
            pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
        elif os.name == 'nt':
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    @staticmethod
    def installTesseract():
        import platform
        if os.name == 'posix':
            system = platform.system()
            if system == 'Linux':
                print("For Linux, you can install Tesseract using your package manager. For example, on Ubuntu, run `sudo apt-get install tesseract-ocr`.")
            elif system == 'Darwin':  # macOS
                print("For macOS, you can install Tesseract using Homebrew. Run `brew install tesseract`.")
            else:
                print("Please download and install Tesseract from: https://github.com/tesseract-ocr/tesseract/wiki/Downloads")
        elif os.name == 'nt':
            print("Please download and install Tesseract from: https://github.com/tesseract-ocr/tesseract/wiki/Downloads")

    @staticmethod
    def preprocessImage(imagePath):
        img = cv2.imread(imagePath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def extractText(imagePath):
        img = OcrProcessor.preprocessImage(imagePath)
        text = pytesseract.image_to_string(img, lang='tur')
        return text

    @staticmethod
    def extractTextWithDetails(imagePath):
        img = OcrProcessor.preprocessImage(imagePath)
        details = pytesseract.image_to_data(img, lang='tur', output_type=Output.DICT)
        return details

    @staticmethod
    def drawBoxes(imagePath, details):
        img = cv2.imread(imagePath)
        nBoxes = len(details['text'])
        for i in range(nBoxes):
            if float(details['conf'][i]) > 60:
                (x, y, w, h) = (details['left'][i], details['top'][i], details['width'][i], details['height'][i])
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img

    def process(self, imagePath):
        try:
            subprocess.check_output(["tesseract", "-v"])
        except subprocess.CalledProcessError:
            self.installTesseract()
        extractedText = self.extractText(imagePath)
        print("Extracted Text:\n", extractedText)
        if self.outputText:
            with open("output.txt", "w") as file:
                file.write(extractedText)
            print("Text saved to 'output.txt'.")
        details = self.extractTextWithDetails(imagePath)
        if self.saveBoxes:
            boxedImage = self.drawBoxes(imagePath, details)
            outputImagePath = "example_image_boxed.jpg"
            cv2.imwrite(outputImagePath, boxedImage)
            print(f"Image with bounding boxes saved as '{outputImagePath}'.")


class PdfProcessor:
    @staticmethod
    def pdfToText(pdfPath, outputTxtPath=None, outputFile=False):
        document = fitz.open(pdfPath)
        text = []
        for pageNum in range(len(document)):
            page = document.load_page(pageNum)
            text.append(page.get_text())
        fullText = "\n".join(text)
        if not outputFile:
            return fullText
        elif outputFile and outputTxtPath is None:
            raise ValueError("'outputTxtPath' must be a string.")
        elif outputFile and outputTxtPath is not None:
            with open(outputTxtPath, "w", encoding="utf-8") as file:
                file.write(fullText)
