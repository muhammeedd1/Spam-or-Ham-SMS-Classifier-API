from pathlib import Path
import joblib

import numpy as np 

from src.config import settings
from src.preprocessing import preprocess

class SpamClassifier :
    def __init__(self , vectorizer_path : str , model_path : str , label_encoder_path : str) -> None:

        paths = [vectorizer_path , model_path , label_encoder_path]

        self._assert_artifacts_exists(*paths)

        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)
    
        

    @staticmethod
    def _assert_artifacts_exists(*paths : str) :
        missing = [path for path in paths if not Path(path).is_file()]

        if missing :
         raise FileNotFoundError(
                "Missing model artifact(s): "
                + ", ".join(missing)
                + ". Did you run `python train.py` first?"
            )

    def predict(self , text : str) :

       cleaned = preprocess(text)

       features = self.vectorizer.transform([cleaned])

       predicted_class = self.model.predict(features)[0]
       probabilities = self.model.predict_proba(features)[0]

       label = self.label_encoder.inverse_transform([predicted_class])[0]

       confidence = float(np.max(probabilities))

       return{
          "label" : label ,
          "is_spam" : label == "spam" ,
          "confidence" : round(confidence , 4),
       }


spamClassifier = SpamClassifier(
   vectorizer_path= settings.TFIDF_VECTORIZER_PATH,
   model_path= settings.MODEL_PATH,
   label_encoder_path=settings.LABEL_ENCODER_PATH,
   
)


