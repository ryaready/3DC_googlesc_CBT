from emotion_classifier import lr_model

lr_model1 = lr_model( 'emotion_classifier_pipeline_lr.pkl')

text = ''
while text != 'quit':
  text = input("Enter a text prompt: ")
  
  if text == 'quit':
    break
    
  else:
    print(lr_model1.predict(text))
