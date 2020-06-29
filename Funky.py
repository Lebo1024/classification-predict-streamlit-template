def file_selector(self):
   file = st.sidebar.file_uploader("train.csv", type="csv")
    if file is not None:
      data = pd.read_csv(file)
      return data
    else:
      st.text("Please upload a csv file")def file_selector(self):
        file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if file is not None:
      data = pd.read_csv(file)
      return data
    else:
      st.text("Please upload a csv file")

#Select Widget
def set_features(self):
   self.features = st.multiselect("Algo type",("Classification","Regression") self.data.columns )

#Data Prep
def prepare_data(self, split_data, train_test):
   # Reduce data size
   data = self.data[self.features]
   data = data.sample(frac = round(split_data/100,2))

# Text Preprocessing
def data_cleaner(input_df):
    train2 = input_df.copy()
    # Removing Twitter Handles (@user)
    def remove_pattern(user_input, pattern):
    
        r = re.findall(pattern, user_input)
    
        for element in r:
            user_input = re.sub(element, "", user_input)
    
        return user_input
    train2["message"] = np.vectorize(remove_pattern)(train2["message"], "@[\w]*")

    # Remove Special Characters,Numbers And Punctuations
    train2["message"] = train2["message"].str.replace("[^a-zA-Z#]", " ") 
    
    # Substituting multiple spaces with single space
    train2["message"] = train2["message"].str.replace(r'\s+', ' ', flags = re.I)
    
    # Converting to Lowercase
    train2["message"] = train2["message"].apply(lambda x: x.lower())

    # Remove Hashtags
    train2["message"] = train2["message"].apply(lambda x: re.compile('\#').sub('', re.compile('rt @').sub('@', x, count=1).strip()))

    # Removing Short Words
    train2["message"] = train2["message"].apply(lambda x: ' '.join([word for word in x.split() if len(word)>3]))

    # Tokenization
    train2["message"] = train2["message"].apply(lambda x: nltk.word_tokenize(x))

    # Remove Stop Words
    def stop_words(user_input):
    
        stop_words = set(stopwords.words('english'))
        wordslist = [word for word in user_input if not word in stop_words]
    
        return wordslist
    train2["message"] = train2["message"].apply(lambda x: stop_words(x))

    # Lemmatization
    stemmer = SnowballStemmer("english")
    train2["message"] = train2["message"].apply(lambda x: [stemmer.stem(word) for word in x])
    
    # Untokenization
    train2["message"] = train2["message"].apply(lambda x: TreebankWordDetokenizer().detokenize(x))
    
    return train2
