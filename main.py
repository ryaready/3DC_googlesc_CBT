from emotion_classifier import lr_model

#Added the print statements because the loading takes awhile
print("Loading model...")
lr_model1 = lr_model( 'emotion_classifier_pipeline_lr.pkl')
print("Done!")
print()

while True:
  text = input("Enter a text prompt: ")
  
  if text == 'quit':
    break
    
  else:
    most_likely_emotion, probability_distribution = lr_model1.predict(text)
    print('Emotion: ', most_likely_emotion)
    print('Emotion Probability Distribution: ', probability_distribution)
    print()
