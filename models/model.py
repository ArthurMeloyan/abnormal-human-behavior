import open_clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image



class Model:
    def __init__(self, settings_path: str = './settings/settings.yaml'):
        with open(settings_path, 'r') as file:
            self.settings = yaml.safe_load(file)

        self.device = self.settings['settings']['device']
        self.model_name = self.settings['settings']['name']
        self.threshold = self.settings['settings']['prediction-threshold']


        # load model and preprocessor
        model_preprocessor = open_clip.create_model_and_transforms(self.model_name, pretrained='openai')
        model = model_preprocessor[0]
        preprocess = model_preprocessor[1]

        self.model = model.to(self.device)
        self.preprocess = preprocess
        self.labels = self.settings['label-settings']['labels']
        self.labels_ = []

        # Generate text features
        for label in self.labels:
            text = label
            self.labels_.append(text)

        self.text_features = self.vectorize_text(self.labels_)
        self.default_label = self.settings['label-settings']['default-label']



    @torch.no_grad()
    def transform_image(self, image: np.ndarray):
        pil_image = Image.fromarray(image).convert('RGB')
        tf_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        return tf_image
    

    @torch.no_grad()
    def tokenize(self, text: list):
        text = open_clip.tokenize(text).to(self.device)
        return text
    

    @torch.no_grad()
    def vectorize_text(self, text: list):
        tokens = self.tokenize(text=text)
        text_features = self.model.encode_text(tokens)
        return text_features
    
    
    @torch.no_grad()
    def predict_(self, text_features: torch.Tensor, image_features: torch.Tensor):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        values, indices = similarity[0].topk(1)
        return values, indices
    

    @torch.no_grad()
    def predict(self, image: np.array) -> dict:
        tf_image = self.transform_image(image)
        image_features = self.model.encode_image(tf_image)
        values, indices = self.predict_(text_features=self.text_features, image_features=image_features)

        label_index = indices[0].cpu().item()
        label_text = self.default_label
        model_confidence = abs(values[0].cpu().item())
        if model_confidence >= self.threshold:
            label_text = self.labels[label_index]
        
        prediction = {'label': label_text, 'confidence': model_confidence}
        return prediction
    

    @staticmethod
    def plot_image(image: np.array, title_text: str):
        plt.figure(figsize=[13, 13])
        plt.title(title_text)
        plt.axis('off')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)