from emotion_classifier import lr_model

# Added the print statements because the loading takes awhile
print("Loading model...")
lr_model1 = lr_model('emotion_classifier_pipeline_lr.pkl')
print("Done!")
print()


# Ideally there will be an intro explaining what's going on plus a disclaimer

def q1():
    wecannot = "Hey, it seems like you're dealing with something that is beyond our scope of healing, we suggest that " \
               "you contact the emergency services in your country "
    ans1 = input("Hey, What happened?, tell us the situation that prompted you to come here:")
    if "suicide" in ans1:
        print(wecannot)
    else:
        most_likely_emotion, probability_distribution = lr_model1.predict(ans1)
        return most_likely_emotion, probability_distribution


question1 = q1()
mle = question1[0]
pd = question1[1]


def q2():
    statement1 = 'It seems like you are feeling a lot of ' + mle + ' ,is that right? (y/n):'
    ans2 = input(statement1)
    correct_emotion = mle
    if "n" in ans2:
        correct_emotion = input("Please tell us what emotion you're feeling right now:")
    else:
        pass
    return correct_emotion


ce = q2()


def q3():
    statement2 = "Can you rate the amount of  " + ce + "  you are feeling on a scale of 10?:"
    ans3 = float(input(statement2))
    emotion_rating = ans3 / 10
    return emotion_rating


er = q3()


# retraining function over here ma'am

def q5():
    statement3 = "What is the main thought that is making you feel this much " + ce + "?:"
    ans4 = input(statement3)
    return ans4


negative_thought = q5()


def q6():
    statement4 = "What is the evidence supporting the idea that " + negative_thought + " is true?:"  # I think we can
    # have a bit more instruction on the HTML side on how to answer each of these questions properly 
    ans5 = input(statement4)
    return ans5


evidence_for = q6()


def q7():
    statement5 = "What is the evidence going against the idea that " + negative_thought + " is true?:"  # again will 
    # req more detail 
    ans6 = input(statement5)
    return ans6


evidence_against1 = q7()


def q8():
    statement6 = "Now, it is important to argue against the reasons that you feel " + negative_thought + " is true."
    statement7 = "To recap, these are the following reasons you believe " + negative_thought + " to be true"
    statement8 = statement6 + evidence_for + statement7
    ans7 = input(statement8)
    return ans7


evidence_against2 = q8()


def q9():
    statement9 = "Great! Now that we have considered all possible arguments, let's re-frame your negative thought, " + negative_thought + ", into a more neutral or positive thought"
    ans8 = input(statement9)
    most_likely_emotion, probability_distribution = lr_model1.predict(ans8)
    statement10 = "Hey, we think you can make a better statement! Try again!:"
    if most_likely_emotion != "neutral":
        while most_likely_emotion != "neutral":
            ans9 = input(statement10)
            most_likely_emotion, probability_distribution = lr_model1.predict(ans9)
    else:
        print("Great, now we can move on to the last step")
    return ans8


more_positive_statement = q9()


def q10():
    statement11 = "and now finally, check in with your emotions, and tell us how much " + ce + "you are feeling right " \
                                                                                               "now, on a scale of 1 " \
                                                                                               "to 10: "
    ans10 = float(input(statement11))
    number = ans10 / 10
    return number


final_emotion_rating = q10()


def q11():
    percentage = (final_emotion_rating / (final_emotion_rating + er)) * 100
    
    statement12 = "you're feeling " + str(percentage) + "better!"
    print(statement12)
    return percentage


final_percentage = q11()

q1()
q2()
q3()
q5()
q6()
q7()
q8()
q9()
q10()
q11()
