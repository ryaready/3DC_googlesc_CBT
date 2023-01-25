from emotion_classifier import lr_model

lr_model1 = lr_model( r'/Users/aishwaryaiyer/Desktop/emotion_classifier_pipeline_lr.pkl')
lr_model1.predict("Slay")
