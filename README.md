# Hoşgeldiniz!

## Phorcys Kütüphanesi Genel Bakış

Phorcys, metin işleme ve analiz için geniş bir işlevsellik yelpazesi sunan kapsamlı bir Türkçe Doğal Dil İşleme (NLP) kütüphanesidir. Medya işlemden metin normalizasyonuna ve model yönetimine kadar çeşitli NLP alanlarını ele alan araçlar içerir. İşte Phorcys kütüphanesinde mevcut olan ana modüllerin genel bir bakışı:

### Kurulum

#### Gereksinimler

Phorcys'i kurmadan önce aşağıdaki gereksinimlerin karşılandığından emin olun:

- Python 3.6 veya daha yüksek bir sürüm
- [ffmpeg](https://www.ffmpeg.org/) indirin
- [ollama](https://ollama.com/) indirin
- [tesseract-ocr](https://tesseract-ocr.github.io/) indirin

#### Kurulum Komutları

```git clone https://github.com/mesutaktss/Phorcys.git```  
```cd Phorcys```  
```pip install -r requirements.txt```

### Datasets

Phorcys ile doğal dil işleme projemizde kullandığımız datasetler:

- **Türkçe Veri Seti**: Türkçe içeriklerle hazırlanmış metin koleksiyonu.
  - [Türkçe büyük veri seti](https://opus.nlpl.eu/results/en&tr-TR/corpus-result-table)
### Özellikler

#### Abyss

- .aby dosyalarını sıkıştırma ve çıkarma işlevleri.

#### Model

- Sınıflandırma modelleri oluşturmak için araçlar.

#### Normalize

- Metin normalizasyonu için işlevler, sayıdan kelimeye dönüştürme ve metin temizleme dahil.

#### ProcessMedia

- Ses dosyalarını metne dönüştürme, altyazı oluşturma ve PDF işleme işlevleri.

#### ProcessText

- Çeşitli metin işleme görevleri için araçlar, normalizasyon ve metin analizi dahil.

#### ProcessWithAI

- Haber analizi, duygu analizi ve çeviri yapma araçları.

#### TextGenerator

- Metin üretme ve oluşturma işlevleri.

---

# Normalize Sınıfı

## Genel Bakış

`Normalize` sınıfı, durak kelimeleri, kötü kelimeleri kaldırma ve sayıları kelimelere dönüştürme gibi metin normalizasyon görevleri için yöntemler sağlar.

## Özellikler

- **numberToWord**: Bir sayıyı Türkçe kelime karşılığına dönüştürür.
- **removeStopwords**: Metinden durak kelimeleri kaldırır.
- **removeBadWords**: Metinden kötü kelimeleri kaldırır ve kaç tane kaldırıldığını sayar.
- **loadStopwords**: Durak kelimeleri bir dosyadan veya verilen bir listeden yükler.

## Kullanım

### Sayıyı Kelimeye Dönüştürme

```Normalize.numberToWord(number)```  
Verilen bir sayıyı Türkçe kelime karşılığına dönüştürür.

### Durak Kelimeleri Kaldırma

```Normalize.removeStopwords(text, stopwords=None)```  
Metinden durak kelimeleri kaldırır. İsteğe bağlı olarak, bir durak kelime listesi sağlayabilirsiniz.

### Kötü Kelimeleri Kaldırma

```Normalize.removeBadWords(text, badwords=None)```  
Metinden kötü kelimeleri kaldırır ve kaç tane kaldırıldığını sayar. İsteğe bağlı olarak, bir kötü kelime listesi sağlayabilirsiniz.

### Durak Kelimeleri Yükleme

```Normalize.loadStopwords(stop_words_source)```  
Durak kelimeleri bir dosyadan veya verilen bir set/listeden yükler.

---

# ProcessText Sınıfı

## Genel Bakış

`ProcessText` sınıfı, metin işleme ve analiz etme için çeşitli yöntemler sunar, bunlar arasında normalizasyon, tokenizasyon ve okunabilirlik analizi bulunmaktadır.

## Özellikler

- **lowerText**: Metni küçük harfe çevirir ve belirli karakter değişiklikleri yapar.
- **upperText**: Metni büyük harfe çevirir ve belirli karakter değişiklikleri yapar.
- **removePunc**: Metinden noktalama işaretlerini kaldırır.
- **accentMarkRemove**: Karakterlerden aksan işaretlerini kaldırır.
- **numToTRText**: Metindeki sayıları Türkçe kelime karşılıklarına dönüştürür.
- **removeNumber**: Metinden sayıları kaldırır.
- **normalize_chars**: Belirli Türkçe karakterleri normalize eder.
- **wordCounter**: Bir cümledeki kelime sayısını hesaplar.
- **wordExtractor**: Bir cümleden kelimeleri çıkarır.
- **sentenceCounter**: Bir paragraftaki cümle sayısını hesaplar.
- **avarageWordCountPerSentence**: Bir cümledeki ortalama kelime sayısını hesaplar.
- **syllableCounter**: Metindeki hece sayısını hesaplar.
- **reabilityTime**: Ortalama okuma hızına göre metni okuma süresini tahmin eder.
- **readabilityScore**: Metnin okunabilirlik skorunu hesaplar.
- **frequencyCalculator**: Metindeki kelime frekanslarını hesaplar.
- **phoneticTransform**: Metni fonetik temsilini dönüştürür.
- **sentenceTokenizer**: Metni cümlelere ayırır.
- **findIdioms**: Metindeki deyimleri, önceden tanımlanmış bir listeye dayanarak bulur.
- **calculateSimilarity**: İki metin arasındaki benzerliği cümle gömme kullanarak hesaplar.
- **deasciify**: ASCII metni Türkçe karakterlere dönüştürür.

## Kullanım

### Metni Küçük Harfe Çevirme

```ProcessText.lowerText(text)```  
Verilen metni küçük harfe çevirir ve belirli karakter değişiklikleri yapar.

### Metni Büyük Harfe Çevirme

```ProcessText.upperText(text)```  
Verilen metni büyük harfe çevirir ve belirli karakter değişiklikleri yapar.

### Noktalama İşaretlerini Kaldırma

```ProcessText.removePunc(text)```  
Verilen metinden noktalama işaretlerini kaldırır.

### Aksan İşaretlerini Kaldırma

```ProcessText.accentMarkRemove(text, accentMap=None)```  
Metindeki karakterlerden aksan işaretlerini kaldırır. İsteğe bağlı olarak, özel bir aksan haritası sağlayabilirsiniz.

### Sayıları Türkçe Metne Çevirme

```ProcessText.numToTRText(text)```  
Metindeki sayıları Türkçe kelime karşılıklarına dönüştürür.

### Sayıları Kaldırma

```ProcessText.removeNumber(text, signed=True, decimal=True)```  
Metinden sayıları kaldırır. İşaretli ve ondalıklı sayılar için seçenekler sunar.

### Karakterleri Normalize Etme

```ProcessText.normalize_chars(text, charTable=None)```  
Metindeki belirli Türkçe karakterleri normalize eder. İsteğe bağlı olarak, özel bir karakter tablosu sağlayabilirsiniz.

### Kelime Sayısı

```ProcessText.wordCounter(sentence)```  
Verilen bir cümledeki kelime sayısını hesaplar.

### Kelimeleri Çıkarma

```ProcessText.wordExtractor(sentence)```  
Verilen bir cümleden kelimeleri çıkarır.

### Cümle Sayısı

```ProcessText.sentenceCounter(paragraph)```  
Verilen bir paragraftaki cümle sayısını hesaplar.

### Cümle Başına Ortalama Kelime Sayısı

```ProcessText.avarageWordCountPerSentence(text)```  
Metindeki cümle başına ortalama kelime sayısını hesaplar.

### Hece Sayısı

```ProcessText.syllableCounter(text)```  
Metindeki hece sayısını hesaplar.

### Okuma Süresi

```ProcessText.reabilityTime(text, avaregeReadTime=190)```  
Ortalama okuma hızına göre metni okuma süresini tahmin eder.

### Okunabilirlik Skoru

```ProcessText.readabilityScore(text)```  
Metnin okunabilirlik skorunu hesaplar ve çeşitli seviyelere ayırır.

### Frekans Hesaplama

```ProcessText.frequencyCalculator(text)```  
Metindeki kelime frekanslarını hesaplar.

### Fonetik Dönüşüm

```ProcessText.phoneticTransform(text)```  
Metni fonetik temsilini dönüştürür.

### Cümle Tokenizasyonu

```ProcessText.sentenceTokenizer(text)```  
Metni cümlelere ayırır.

### Deyimleri Bulma

```ProcessText.findIdioms(text)```  
Metindeki deyimleri, önceden tanımlanmış bir listeye dayanarak bulur.

### Benzerlik Hesaplama

```ProcessText.calculateSimilarity(text1, text2, modelName="dbmdz/bert-base-turkish-cased")```  
İki metin arasındaki benzerliği cümle gömme kullanarak hesaplar.

### Deasciify

```ProcessText.deasciify(text)```  
ASCII metni Türkçe karakterlere dönüştürür.

---

# ProcessWithAI Sınıfı

## Genel Bakış

`ProcessWithAI` sınıfı, çeşitli metin analiz görevlerini AI tabanlı yöntemlerle gerçekleştirir. Bu yöntemler arasında haber analizi, duygu analizi, çeviri ve daha fazlası bulunur.

## Özellikler

- **NewsAnalysis**: Metin haber analizi yapar.
- **DetailedEmotionAnalysis**: Metnin detaylı duygu analizini yapar.
- **SimpleEmotionAnalysis**: Metnin basit duygu analizini yapar.
- **BullyingAnalysis**: Metinde zorbalık olup olmadığını analiz eder.
- **TranslateToTurkish**: Metni İngilizce'den Türkçe'ye çevirir.
- **TranslateToEnglish**: Metni Türkçe'den İngilizce'ye çevirir.
- **ImageToText**: Görüntüleri metne dönüştürür.
- **NERAnalysis**: Adlandırılmış varlık tanıma yapar.
- **EntityBasedSentimentAnalysis**: Varlık bazlı duygu analizi yapar.
- **generateText**: Yapay zeka ile metin üretir.
- **generateResponse**: Metne yanıt oluşturur.
- **createDummyText**: Rastgele metin oluşturur.
- **generateClassificationModel**: Sınıflandırma modeli oluşturur.
- **PDFProcessor**: PDF belgelerini metne dönüştürür.
- **OCRProcessor**: Görüntülerden metin çıkarır.
- **MediaToText**: Ses ve video dosyalarını metne dönüştürür.
- **Normalize**: Metni normalleştirir.
- **ProcessText**: Basit metin işlemleri yapar.

## Kullanım

### Haber Analizi

```ProcessWithAI.NewsAnalysis(text)```  
Verilen metni haber analizi yapar.

### Detaylı Duygu Analizi

```ProcessWithAI.DetailedEmotionAnalysis(text)```  
Metnin detaylı duygu analizini yapar.

### Basit Duygu Analizi

```ProcessWithAI.SimpleEmotionAnalysis(text)```  
Metnin basit duygu analizini yapar.

### Zorbalık Analizi

```ProcessWithAI.BullyingAnalysis(text)```  
Metinde zorbalık olup olmadığını analiz eder.

### Çeviri (Türkçe'ye)

```ProcessWithAI.TranslateToTurkish(text)```  
İngilizce metni Türkçe'ye çevirir.

### Çeviri (İngilizce'ye)

```ProcessWithAI.TranslateToEnglish(text)```  
Türkçe metni İngilizce'ye çevirir.

### Görüntüden Metin

```ProcessWithAI.ImageToText(image)```  
Görüntüyü metne dönüştürür.

### Adlandırılmış Varlık Tanıma

```ProcessWithAI.NERAnalysis(text)```  
Metinde adlandırılmış varlıkları tanır.

### Varlık Bazlı Duygu Analizi

```ProcessWithAI.EntityBasedSentimentAnalysis(text)```  
Varlık bazlı duygu analizi yapar.

### Metin Üretimi

```ProcessWithAI.generateText(prompt)```  
Yapay zeka ile metin üretir.

### Yanıt Oluşturma

```ProcessWithAI.generateResponse(prompt)```  
Metne yanıt oluşturur.

### Rastgele Metin Oluşturma

```ProcessWithAI.createDummyText(length)```  
Rastgele uzunlukta metin oluşturur.

### Sınıflandırma Modeli Oluşturma

```ProcessWithAI.generateClassificationModel(data)```  
Sınıflandırma modeli oluşturur.

### PDF İşleme

```ProcessWithAI.PDFProcessor(pdf_file)```  
PDF dosyasını metne dönüştürür.

### OCR İşleme

```ProcessWithAI.OCRProcessor(image)```  
Görüntülerden metin çıkarır.

### Medyayı Metne Dönüştürme

```ProcessWithAI.MediaToText(media_file)```  
Ses ve video dosyalarını metne dönüştürür.

### Metin Normalizasyonu

```ProcessWithAI.Normalize(text)```  
Metni normalleştirir.

### Basit Metin İşlemleri

```ProcessWithAI.ProcessText(text)```  
Basit metin işlemleri yapar.

### Lisans

Phorcys ve bağlı bileşenler, [Affero Genel Kamu Lisansı (AGPL)](https://www.gnu.org/licenses/agpl-3.0.html) altında lisanslanmıştır.
