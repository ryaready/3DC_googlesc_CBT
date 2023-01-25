#Load Exploratory Data Analysis (EDA) Packages
import pandas as pd

# Load Text Cleaning Packages
import neattext.functions as nfx

# Load Machine Learning Packages
from sklearn.linear_model import LogisticRegression

#Load transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#To Build Pipeline
from sklearn.pipeline import Pipeline

#Load saving/loading tool
import joblib

#Logistic regression model class
class lr_model():
    #The pipeline filename refers to the exisiting model.
    #Filename must include the extension .pkl
    def __init__(self, pipeline_filename):
        self.pipeline_filename = pipeline_filename


    def predict(self, text_to_predict):
        #Loads in the existing model and predicts
        pipeline = joblib.load(self.pipeline_filename)
        
        #This returns the highest probability emotion as a list of one element
        # eg. ['anger']
        prediction = pipeline.predict([text_to_predict])
        prediction = prediction[0] #remove the list

        #This will give the probability of all emotions
        #They correspond to each other
        emotions = pipeline.classes_
        probability = pipeline.predict_proba([text_to_predict])[0] #Its a nested list for no reason again

        #Pair them together
        emotions_probabilities = list(zip(emotions, probability))

        #Return both, up to you how you want to use them
        return (prediction, emotions_probabilities)

    #RETRAINING WILL TAKE ABOUT A MINUTE SO NOT SURE IF YOU WANT TO DO THIS IN THE MIDDLE OF THE CONVO
    #Pipeline_filename here is the new filename to save the new model under
    #The filename MUST HAVE .pkl as extension
    #Use same name to override old model
    #new_text_dict is a DICTIONARY of {"Emotion: <correct emotion>, "Text": <text to add>}
    def retrain(self, new_text_dict, new_pipeline_filename):
        #Load existing dataset to dataframe
        df = pd.read_csv("emotion_dataset_2.csv") #Change if we are using different dataset later on

        #Add new text as a new entry to database
        df.append(new_text_dict)

        #Remove user handles
        #Will override the Clean_text collumn
        df["Clean_Text"] = df['Text'].apply(nfx.remove_userhandles) #remove @whatever etc

        #Remove stopwords (common words)
        df["Clean_Text"] = df['Clean_Text'].apply(nfx.remove_stopwords)

        #Remove special character (anything not alphabet or numbers)
        df["Clean_Text"] = df['Clean_Text'].apply(nfx.remove_special_characters)


        # Get Features and Labels
        Xfeatures = df['Clean_Text']
        ylabels = df['Emotion']

        #Split Data
        x_train, x_test, y_train, y_test = train_test_split(Xfeatures, ylabels, test_size=0.2, random_state=42)

        #LogisticRegression Pipeline
        pipe_lr = Pipeline(steps=[('cv', CountVectorizer()),
                                #('tfidf', TfidfTransformer()),
                                ('lr', LogisticRegression(max_iter=300, class_weight='balanced'))])
        pipe = pipe_lr

        # Train and Fit Data
        # Yes this is the training!
        print("Starting training...")
        pipe.fit(x_train, y_train)

        #Save model and pipeline
        with open(new_pipeline_filename, 'wb') as pipeline_file:
            joblib.dump(pipe, pipeline_file)

        print("Done!") #Just for debugging if you need it

lr_model1 = lr_model( r'/Users/aishwaryaiyer/Desktop/emotion_classifier_pipeline_lr.pkl')
lr_model1.predict("Slay")
