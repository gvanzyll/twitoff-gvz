from .models import User
from sklearn.linear_model import LogisticRegression
import numpy as np
from .twitter import vectorize_tweet


def predict_user(user0_username, user1_username, hypo_tweet_text):

    user0 = User.query.filter(User.username == user0_username).one()
    user1 = User.query.filter(User.username == user1_username).one()

    user0_vects = np.array([tweet.vect for tweet in user0.tweets])
    user1_vects = np.array([tweet.vect for tweet in user1.tweets])

    # Combine the two user's word embeddings
    vects = np.vstack([user0_vects, user1_vects])

    # Create a np array for zeroes and ones
    zeroes = np.zeros(len(user0.tweets))
    ones = np.ones(len(user1.tweets))

    labels = np.concatenate([zeroes, ones])

    # Import and Train our logistic regression
    log_reg = LogisticRegression()

    # Train our logistic regression
    log_reg.fit(vects, labels)

    # Get the word embeddings for our hypo_tweet_text
    hypo_tweet_vect = vectorize_tweet(hypo_tweet_text)

    # Generate a prediction
    prediction = log_reg.predict([hypo_tweet_vect])

    return prediction[0]
