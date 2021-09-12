### Libraries needed

keras 2.4.3
gensim 3.8.3
matplotlib 3.1.2
nltk 3.5
numpy 1.16.2
oauthlib 3.1.0
pandas 0.25.3
pickleshare 0.7.5
pymed 0.5.1
regex 2020.7.14
requests 2.22.0
scapy 2.4.2
scikit-learn 0.22.1
scipy 1.4.1
seaborn 0.9.0
sent2vec 0.1.6
spacy 2.3.2
six 1.13.0
tensorflow 2.3.0
textblob 0.15.3


### How to run?

Main function contains a variable text where the text that needs to be compared to the dataset is added.
In the Tests part in main, each commented line can be uncommented and run. Each function call will call the 
necessary algorithms and the result is saved in "results" folder.


### Extra datasets needed

nltk - can be installed by running "nltk.download()" command <br/>
Google pre-trained vector model - can be installed by running "svn checkout http://word2vec.googlecode.com/svn/trunk/" command <br/>
en_core_web_sm - can be downloaded by running "python -m spacy download en_core_web_sm" command 
