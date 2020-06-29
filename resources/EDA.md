'''#Exploratory Data Analysis

##Train Dataset

Train Dataset Description

train.describe().astype(int)

#Clean Vs Unclean Data

unclean_tweets = train["message"].str.len()
clean_tweets = train_df["message"].str.len()



plt.hist(unclean_tweets, label = 'Uncleaned_Tweet')
plt.hist(clean_tweets, label = 'Cleaned_Tweet')
plt.legend()
plt.show()
'''
