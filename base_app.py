"""
    Simple Streamlit webserver application for serving developed classification
	models.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------
    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.
	For further help with the Streamlit framework, see:
	https://docs.streamlit.io/en/latest/
"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")



#variables definition
variables = pd.DataFrame(["sentiment", "message", "tweetid"], columns = ["variables"])
variables["definition"] = pd.DataFrame(["sentiment of twitter messages", "twitter messages", "twitter unique id"])
variables.head()

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

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifier")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "Exploratory Data Analysis(EDA)"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

	# Building out the "Exploratory Data Analysis(EDA)" page
	if selection == "Exploratory Data Analysis(EDA)":
		st.info("See the EDA for climate Change tweets")
		# You can read a markdown file from supporting resources folder
		model = open("resources/EDA.md","r")

		st.markdown(model.read())

		raw.describe().astype(int)

		unclean_tweets = train["message"].str.len()
		clean_tweets = train_df["message"].str.len()

		plt.hist(unclean_tweets, label = 'Uncleaned_Tweet')
		plt.hist(clean_tweets, label = 'Cleaned_Tweet')
		plt.legend()
		plt.show()

# Building out the "Exploratory Data Analysis(EDA)" page
	if selection == "Classification":
		def set_classifier_properties(self):
			self.type = st.sidebar.selectbox("Algorithm type", ("Classification", "Regression"))
			if self.type == "Regression" :
      		
			self.chosen_classifier = st.sidebar.selectbox(“Please choose a classifier”, (‘Random Forest’, ‘Linear Regression’, ‘NeuralNetwork’))

			if self.chosen_classifier == ‘Random Forest’:
			   self.n_trees = st.sidebar.slider(‘number of trees’, 1, 1000, 1)
			elif self.chosen_classifier == ‘Neural Network’:
			   self.epochs = st.sidebar.slider(‘number of epochs’, 1 ,100 ,10)
			   self.learning_rate = float(st.sidebar.text_input(‘learning rate:’, ‘0.001’))
			elif self.type == “Classification”:
			   self.chosen_classifier = st.sidebar.selectbox(“Please choose a classifier”, (‘Logistic Regression’, ‘Naive Bayes’, ‘Neural Network’))
			if self.chosen_classifier == ‘Logistic Regression’:
			   self.max_iter = st.sidebar.slider(‘max iterations’, 1, 100, 10)
			elif self.chosen_classifier == ‘Neural Network’:
			   self.epochs = st.sidebar.slider(‘number of epochs’, 1 ,100 ,10)
			   self.learning_rate = float(st.sidebar.text_input(‘learning rate:’, ‘0.001’))
			   self.number_of_classes = int(st.sidebar.text_input(‘Number of classes’, ‘2’))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
