from django.shortcuts import render
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import io
from PIL import Image
#cd django-dev
#source my_env/bin/activate
#alias python=python3
#python manage.py runserver localhost:8000
#abra o firefox e digite localhost:8000/MachineLearning

class ControllerML(): #tratar com scikitlearning

  corpus = []
  labels = []
  
  X = []#Recebe o corpus
  Y = []#recebe labels
  
  def __init__(self):
    self.Brasil = []
    self.Mundo = []
    self.Economia = []
    self.CienciaSaude = []
    self.Politica = []
    self.Blog = []
    self.Cultura = []
    
  def adicionarNoticiaBrasil(self,n):
    self.Brasil.append(n)
  def adicionarNoticiaMundo(self,n) :
    self.Mundo.append(n)
  def adicionarNoticiaEconomia(self,n):
    self.Economia.append(n)
  def adicionarNoticiaCienciaSaude(self,n):
    self.CienciaSaude.append(n)
  def adicionarNoticiaPolitica(self,n) :
    self.Politica.append(n)
  def adicionarNoticiaBlog(self,n):
    self.Blog.append(n)
  def adicionarNoticiaCultura(self,n):
    self.Cultura.append(n)
       
  def criarCorpus(self):
    #for noticia in self.Brasil:
     # self.corpus.append(noticia)
      #self.labels.append('Brasil')
    #for noticia in self.Mundo:
     # self.corpus.append(noticia)
      #self.labels.append('Mundo')
    #for noticia in self.Economia:
      #self.corpus.append(noticia)
      #self.labels.append('Economia')
    for noticia in self.CienciaSaude:
      self.corpus.append(noticia)
      self.labels.append('Ciência e Saúde')
    for noticia in self.Politica:
      self.corpus.append(noticia)
      self.labels.append('Politica')
    #for noticia in self.Blog:
     # self.corpus.append(noticia)
      #self.labels.append('Blog')
    #for noticia in self.Cultura:
     # self.corpus.append(noticia)
      #self.labels.append('Cultura')
    self.X = self.corpus
    self.Y = self.labels
    return self.corpus
    
  def evalueteModel(self, Y_test, Y_pred_rv):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    sns.set(font_scale=0.8) 
    cm = confusion_matrix(Y_test, Y_pred_rv)
    sns.heatmap(cm, xticklabels=['predicted_CienciaSaude', 'predicted_Politica'], yticklabels=['actual_CienciaSaude', 'actual_Politica'],annot=True, fmt='d', annot_kws={'fontsize':20}, cmap="YlGnBu");
    true_neg, false_pos = cm[0]
    false_neg, true_pos = cm[1]
    accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg),3)
    precision = round((true_pos) / (true_pos + false_pos),3)
    recall = round((true_pos) / (true_pos + false_neg),3)
    f1 = round(2 * (precision * recall) / (precision + recall),3)
    #image_file =plt.savefig('heatmap.png', format= 'png') 
    plt.show()
    #image_file = plt.savefig('fig.png', dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format='png',transparent=True, bboxinches=None, padinches=0., frameon=None)
    s = ''
    s = s + 'Accuracy: {}'.format(accuracy)
    s = s + 'Precision: {}'.format(precision)
    s = s + 'Recall: {}'.format(recall)
    s = s + 'F1 Score: {}'.format(f1)
    return(s)

  def NaiveBayesEvaluete(self, X_train_rv, X_test_rv, Y_train, Y_test, st): 
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    # Train the model
    nb.fit(X_train_rv, Y_train)
    # Take the model that was trained on the X_train_cv data and apply it to the X_test_cv
    #data
    Y_pred_rv = nb.predict(X_test_rv)
    plt.suptitle("Evaluete Model: " + st, color='m')
    #print(Y_pred_cv_nb)
    return self.evalueteModel(Y_test, Y_pred_rv)

  def logistRegressionEvaluete(self, X_train_rv, X_test_rv,Y_train, Y_test, st):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    # Train the model
    lr.fit(X_train_rv, Y_train)
    # Take the model that was trained on the X_train_cv data and apply it to the X_test_cv
    #data
    Y_pred_rv = lr.predict(X_test_rv)
    #Input:
    plt.suptitle("Evaluete Model: " + st, color='m')
    #print(Y_pred_rv)
    return self.evalueteModel(Y_test, Y_pred_rv)

  #rv = representacao vetorial SVM Kernel=Linear
  def SvmLinearSVCEvaluete(self, X_train_rv, X_test_rv,Y_train, Y_test, st):
    from sklearn.svm import LinearSVC
    sl = LinearSVC()
    sl.fit(X_train_rv, Y_train)
    Y_pred_rv = sl.predict(X_test_rv)
    plt.suptitle("Evaluete Model: " + st, color='m')
    #print(Y_pred_rv)
    return self.evalueteModel(Y_test, Y_pred_rv)

  def extrairTemaTopico(self,topico, v):
    tema = ''
    for indexword in topico:
      tema = tema + v.get_feature_names()[indexword] + ' '
    return tema

  def rotular(self, vDescriptions, posNoticia):
    from sklearn.decomposition import PCA
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer

    from sklearn.decomposition import NMF
    tfidf = TfidfVectorizer(max_features= 1000, max_df = 0.5, smooth_idf=True)
    X = tfidf.fit_transform(vDescriptions)
    nmf = NMF(n_components=10, random_state=42)
    nmf.fit_transform(X[posNoticia])
    #topicos = extrairTopicos(nmf)
    #data.shape
    #print(data[0])
    #print('\nDocumento', i)
    sorted_0 = nmf.components_[0].argsort() #primeiro tópico
    #sorted_1 = pca.components_[1].argsort() #segundo tópico
    #sorted_2 = pca.components_[2].argsort() #terceiro tópico
    sortedflip_0 = np.flip(sorted_0)[0:5]
    #sortedflip_1 = np.flip(sorted_1)[0:5]
    #sortedflip_2 = np.flip(sorted_2)[0:5]
    #print('Tópico 1:')
    return self.extrairTemaTopico(sortedflip_0, tfidf)
   # print('Tópico 2:')
    #extrairTemaTopico(sortedflip_1, cv)
    #print('Tópico 3:')
    #extrairTemaTopico(sortedflip_2, cv)


  def treinarMachineLearning(self):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from itertools import combinations
    from sklearn.metrics.pairwise import cosine_similarity


      # split the data into a training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.3, random_state=42)
    # test size = 30% of observations, which means training size = 70% of observations
    # random state = 42, so we all get the same random train / test split
    #cv = CountVectorizer(stop_words=‘english’)
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    X_train_cv = cv.fit_transform(X_train) # fit_transform learns the vocab and one-hot encodes
    X_test_cv = cv.transform(X_test) # transform uses the same vocab and one-hot encodes
    
    #Tfidf    
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf = TfidfVectorizer()
    X_train_tf = tf.fit_transform(X_train) # fit_transform learns the vocab and one-hot encodes
    X_test_tf = tf.transform(X_test) # transform uses the same vocab and one-hot encodes

    
    # Use a logistic regression model CV
    rl_cv = self.logistRegressionEvaluete(X_train_cv, X_test_cv,Y_train, Y_test, 'Logistic regression model CV')
    # Use a logistic regression model tfidf
    rl_tf = self.logistRegressionEvaluete(X_train_tf, X_test_tf,Y_train, Y_test, 'Logistic regression model tfidf')
    # Use a Naive Bayes model CV
    nb_cv = self.NaiveBayesEvaluete(X_train_cv, X_test_cv,Y_train, Y_test, 'Naive Bayes model CV')
    # Use a Naive Bayes model tfidf
    nb_tf = self.NaiveBayesEvaluete(X_train_tf, X_test_tf,Y_train, Y_test, 'Naive Bayes model tfidf')
    # Use a SVM Linear SVC LINEAR model CV
    svm_cv = self.SvmLinearSVCEvaluete(X_train_cv, X_test_cv,Y_train, Y_test, 'SVM Linear SVC LINEAR model CV')
    # Use a SVM Linear SVC LINEAR model tfidf
    svm_tf = self.SvmLinearSVCEvaluete(X_train_tf, X_test_tf,Y_train, Y_test, 'SVM Linear SVC LINEAR model tfidf')
    return (rl_cv,rl_tf, nb_cv,nb_tf, svm_cv, svm_tf)
    

from django.http import HttpResponse
import re
import string
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


import json

from pyUFbr.baseuf import ufbr

#Categoria: Brasil, Mundo, Economia, Ciência e Saúde, Política, Blog, Cultura,

estados_do_brasil = ['Acre','Alagoas','Amapá','Amazonas','Bahia','Ceará','Distrito Federal','Espirito Santo','Goiás','Maranhão','Mato Grosso do Sul','Mato Grosso','Minas Gerais','Pará','Paraíba','Paraná','Pernambuco','Piauí','Rio de Janeiro','Rio Grande do Norte','Rio Grande do Sul','Rondônia','Roraima','Santa Catarina','São Paulo','Sergipe','Tocantins'];

cidades_sp = ufbr.list_cidades('SP')

Noticias             = []
vDescriptions        = []

def preparaTexto(text):
  if text == None : text = '' #puxar tudo abrir o link
  clean_text = re.sub('\w*\d\w*',' ',text) #elimina palavras com numeros
  clean_text = re.sub('[%s]' % re.escape(string.punctuation),' ', clean_text.lower()) #elimina pontuacao e torna o texto minusculo
  tokens = word_tokenize(clean_text)#tokeniza
  tokens_without_sw = [word for word in tokens if not word in stopwords.words('portuguese')]
  text_without_sw = TreebankWordDetokenizer().detokenize(tokens_without_sw) #retira stop words
  return(text_without_sw)

def aprendizagem_ns(ml,Noticias, vDescriptions):  
  i = 0 
  html = '' 
  for noticia in Noticias:
    html = html + '<a href="'+ noticia[0] +'">' + noticia[1] + '</a>' + '(' + noticia[2].strip() + ')'+ '(' + ml.rotular(vDescriptions, i) + ')' +' <br>'
    i+=1
  return html

def carrega(noticias, ml):#, flag):
  html = ''
  pol = 0
  cien = 0
  for noticia in noticias:
    if (noticia['link'] != None):
      if noticia['categoria'] == None : noticia['categoria'] = 'Não Categorizada'
      if noticia['title'] == None : noticia['title'] = 'Sem titulo'
      texto = preparaTexto(noticia['description'])
      if noticia['categoria'].strip().upper().find('MUNDO') != -1  : ml.adicionarNoticiaMundo(texto)
      if noticia['categoria'].strip() in estados_do_brasil or noticia['categoria'].strip().upper() in cidades_sp : 
    	  noticia['categoria'] = 'Brasil'
    	  ml.adicionarNoticiaBrasil(texto)	
      if noticia['categoria'].strip().find('Blog') != -1 : 
    	  noticia['categoria'] = 'Blog'
    	  ml.adicionarNoticiaBlog(texto)	
      if noticia['categoria'].strip().upper().find('CORONAVÍRUS') != -1 or noticia['categoria'].strip().upper().find('VÍRUS') != -1 or noticia['categoria'].strip().upper().find('VACINA') != -1: 
        noticia['categoria'] = 'Ciência e Saúde'
        ml.adicionarNoticiaCienciaSaude(texto)
        cien += 1
        Noticias.append([noticia['link'],noticia['title'],noticia['categoria'],texto])
        vDescriptions.append(texto)
      if noticia['categoria'].strip().upper().find('CONCURSO') != -1 or noticia['categoria'].strip().upper().find('EMPREGO') != -1 or  noticia['categoria'].strip().upper().find('IMPOSTO') != -1  or noticia['categoria'].strip().upper().find('NEGÓCIO') != -1 or noticia['categoria'].strip().upper().find('FINANCEIRA') != -1 : 
        noticia['categoria'] = 'Economia'
        ml.adicionarNoticiaEconomia(texto)	
      if noticia['categoria'].strip().find('Pop & Arte') != -1 or noticia['categoria'].strip().upper().find('MÚSICA') != -1 or 	noticia['categoria'].strip().upper().find('CINEMA') != -1 or noticia['categoria'].strip().upper().find('LIVES') != -1 or noticia['categoria'].strip().find('Agora é Assim?') != -1 or noticia['categoria'].strip().upper().find('CARNAVAL') != -1  : 
        noticia['categoria'] = 'Cultura' 
        ml.adicionarNoticiaCultura(texto) 
      if noticia['categoria'].strip().find('Política') != -1	:
         ml.adicionarNoticiaPolitica(texto)
         pol += 1

      #if flag == 'NS':      
         Noticias.append([noticia['link'],noticia['title'],noticia['categoria'],texto])
         vDescriptions.append(texto)
      
      #if flag == 'S': 
      html = html + '<a href="'+ noticia['link'] +'">' + noticia['title'] + '</a>' + '(' + noticia['categoria'].strip() + ')' +' <br>'
  
  #if flag == 'NS':
  html = aprendizagem_ns(ml,Noticias, vDescriptions)

  html = html + 'Noticias de Politica: ' + str(pol) + ' Noticias de Ciência e Saúde: ' + str(cien) + '\n<br>'
  return html

def carregaS(request,noticias):
  ml = ControllerML()
  html = carrega(noticias, ml)#, 'S') #noticias string do arquivo json
  ml.criarCorpus()
  rl_cv,rl_tf, nb_cv,nb_tf, svm_cv, svm_tf = ml.treinarMachineLearning()
  dados1 = 'Regressão Log CV: '+rl_cv
  dados2 = 'Regressão Log tfidf: '+rl_tf
  dados3 = 'Naive Bayes com CV: '+nb_cv
  dados4 = 'Naive Bayes com tfidf: '+nb_tf
  dados5 = 'smv K=LINEAR com CV: '+svm_cv
  dados6 = 'smv K=LINEAR com tfidf: '+svm_tf
  html = html + dados1 + '<br>' + dados2 + '<br>' + dados3 + '<br>' + dados4 + '<br>' + dados5 + '<br>' + dados6 + '<br>'
  return html

def carregaNS(request,noticias):
  ml = ControllerML()
  html = carrega(noticias, ml)#, 'NS') #noticias string do arquivo json
  return html

# Create your views here. 
def index(request):
  with open('noticias.json', 'r') as json_file:	
    noticias = json.load(json_file)
  
  #html = '<a href="'+carregaNS(request,noticias) + '">' + 'Carregar Não Supervisionado' + '</a>  <br>\n'
  #html = html + '<a href="'+ carregaS(request,noticias) + '">' + 'Carregar Supervisionado' + '</a>  <br>'
  
  #Brasil, Mundo, Economia, CienciaSaude,Politica, Blog, Cultura
  ml = ControllerML()
  html = carrega(noticias, ml) #noticias string do arquivo json
  ml.criarCorpus()
  rl_cv,rl_tf, nb_cv,nb_tf, svm_cv, svm_tf = ml.treinarMachineLearning()
  #fig1 = rl_cv[0]
  dados1 = 'Regressão Log CV: '+rl_cv
  dados2 = 'Regressão Log tfidf: '+rl_tf
  dados3 = 'Naive Bayes com CV: '+nb_cv
  dados4 = 'Naive Bayes com tfidf: '+nb_tf
  dados5 = 'smv K=LINEAR com CV: '+svm_cv
  dados6 = 'smv K=LINEAR com tfidf: '+svm_tf
  html = html + dados1 + '<br>' + dados2 + '<br>' + dados3 + '<br>' + dados4 + '<br>' + dados5 + '<br>' + dados6 + '<br>'
  return HttpResponse(html)
