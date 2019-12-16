# Textmining 

## Instalación
```
import nltk
nltk.download('punkt')
```

### Sólo uso:

```
   cd textmining/
   python setup.py install
```

### Modo de desarrollo:

```   
   pip install -e textmining
```
## Archivos


* Clustering.py  -> sólo calcula main coherencia ?¡¡


* ContentAnalysis.py

- clean_content 

- cleaning_tokenize

- cleaning_stopwords

- get_tokens_text

- get_stopwords


* TextMining.py



funciones:

- calculate_words_frecuencies -> devuelve la cantidad de n palabras más frecuentes, recibe un parámetro n. 

- main_calculate_bigram_frecuencies ->  devuelve los bigramas más frecuentes llamando a la funcion calculate_bigram_frecuencies tomando de un dataframe el campo "COMMENT", que son los comentarios de los reportes "otro". Se podría parametrizar cual es el campo de texto.

- get_word_cloud -> calcula la nube de palabras.


- get_best_kmeans_k_parameter ->  calcula el mejor k para kmedias

- get_best_btm_k_parameter -> calcula el mejor k para btm

- get_best_k_parameter_lda ->  calcula el mejor k para lda y llama a la función compute_coherence_values


* BTM.py







