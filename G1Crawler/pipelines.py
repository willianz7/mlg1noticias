# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import re
from itemadapter import ItemAdapter
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.preprocessing import StandardScaler
#nltk.download('all')
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.tag import pos_tag
#!pip install nltk

nltk.download('rslp')
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize.treebank import TreebankWordDetokenizer


class G1CrawlerPipeline:
  # yield({'title':title, 'description':description, 'image_url':image_url, 'link':link,'categoria':cat})

    def preparaTexto(text):
        clean_text = re.sub('\w*\d\w*',' ',text) #elimina palavras com numeros
        clean_text = re.sub('[%s]' % re.escape(string.punctuation),' ', clean_text.lower()) #elimina pontuacao e torna o texto minusculo
        #text_tokens = word_tokenize(clean_text)
        #tokens_without_sw = [word for word in tokens if not word in stopwords.words('portuguese')]
        #text_without_sw = TreebankWordDetokenizer().detokenize(tokens_without_sw)
        return(text_tokens)
        #return(tokens_without_sw)

    
    def process_item(self, item, spider):
        item['title'] = item['title'].lower()
        description = preparaTexto(item['description'].lower())
        item['categoria'] = item['categoria'].lower()
        print(item)
        line =  json.dumps(dict(item)) + '\n'
        self.file.write(line)
        return item
