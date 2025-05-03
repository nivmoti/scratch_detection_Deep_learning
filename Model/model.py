import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import zipfile
from datetime import datetime
import shutil
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from Model.unet import ImprovedUNet
from datamap.Waferclass import WaferDataset
from tqdm import tqdm


class Model():
    def __init__(self,df_train,model_dir = "Model/SaveModel"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.df_train = df_train
        self.model = ImprovedUNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0], device=self.device))


    def __create_wafer_map(self,df, wafer_name, col, fill=0):
        wafer_df = df[df['WaferName'] == wafer_name]
        min_x, max_x = wafer_df['DieX'].min(), wafer_df['DieX'].max()
        min_y, max_y = wafer_df['DieY'].min(), wafer_df['DieY'].max()
        width, height = max_x - min_x + 1, max_y - min_y + 1
        wafer_map = np.full((height, width), fill, dtype=np.float32)

        for _, row in wafer_df.iterrows():
            x, y = int(row['DieX'] - min_x), int(row['DieY'] - min_y)
            wafer_map[y, x] = row[col]
        return wafer_map

    def __build_wafer_cache(self,df, wafer_list, cache_dir, label_col='IsScratchDie', is_test=False):
        os.makedirs(os.path.join(cache_dir, "images"), exist_ok=True)
        if not is_test:
            os.makedirs(os.path.join(cache_dir, "labels"), exist_ok=True)

        for name in wafer_list:
            img_path = os.path.join(cache_dir, "images", f"{name}.pt")

            if not os.path.exists(img_path):
                wafer_df = df[df['WaferName'] == name]
                min_x, max_x = wafer_df['DieX'].min(), wafer_df['DieX'].max()
                min_y, max_y = wafer_df['DieY'].min(), wafer_df['DieY'].max()

                # Create blank 256x256 grid
                image = torch.zeros((1, 256, 256), dtype=torch.float32)

                for _, row in wafer_df.iterrows():
                    x = int(row['DieX'] - min_x)
                    y = int(row['DieY'] - min_y)
                    if 0 <= x < 256 and 0 <= y < 256:
                        image[0, y, x] = row['IsGoodDie']

                torch.save(image, img_path)

            if not is_test:
                lbl_path = os.path.join(cache_dir, "labels", f"{name}.pt")
                if not os.path.exists(lbl_path):
                    wafer_df = df[df['WaferName'] == name]

                    label = torch.zeros((256, 256), dtype=torch.long)

                    for _, row in wafer_df.iterrows():
                        x = int(row['DieX'] - min_x)
                        y = int(row['DieY'] - min_y)
                        if 0 <= x < 256 and 0 <= y < 256:
                            label[y, x] = row[label_col]

                    torch.save(label, lbl_path)

        print(" Fixed-coordinate cache built.")
    def makeData(self):
        """
        Convert the train dataset into train and test dataset in a form of a image.
        :return: data dir containing the train dataset as a image form.
        """
        wafer_names = self.df_train['WaferName'].unique()
        self.__build_wafer_cache(self.df_train, wafer_names, cache_dir=os.path.join('datamap', 'data', 'cache'))

    def __train_model(self,model, train_loader, val_loader, optimizer, criterion, device,
                    num_epochs=30, save_path=os.path.join('Model','SaveModel', "best_model_new.pth"),
                    loss_path=os.path.join('Model','SaveModel', "best_val_loss_new.txt"),
                    opt_path=os.path.join('Model','SaveModel', "best_optimizer_new.pth"),
                    continue_training=True):
        """
        Train loop for the model
        :param model: a deep learning model
        :param train_loader: loader of training data
        :param val_loader: loader of validation data
        :param optimizer: optimizer for the model
        :param criterion: criterion for the model
        :param device: torch device
        :param num_epochs:
        :param save_path: save path
        :param loss_path: loss path
        :param opt_path: optimizer path
        :param continue_training: boolean flag to continue training or not
        :return:
        """

        best_val_loss = float('inf')
        start_epoch = 0

        # --- Try to load existing model, best_val_loss, optimizer ---
        if continue_training:
            if os.path.exists(save_path):
                model.load_state_dict(torch.load(save_path))
                print(f" Loaded model from {save_path}")
            if os.path.exists(loss_path):
                with open(loss_path, 'r') as f:
                    best_val_loss = float(f.read().strip())
                print(f"Loaded best validation loss: {best_val_loss:.4f}")
            if os.path.exists(opt_path):
                optimizer.load_state_dict(torch.load(opt_path))
                print(f"Loaded optimizer state from {opt_path}")

        early_stopping_counter = 0
        patience = 3

        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # --- Validation ---
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # --- Save best model + loss + optimizer ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                with open(loss_path, 'w') as f:
                    f.write(str(best_val_loss))
                torch.save(optimizer.state_dict(), opt_path)
                early_stopping_counter = 0
                print(f" Best model and optimizer saved at Epoch {epoch + 1} with Val Loss: {val_loss:.4f}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f" Early stopping triggered at Epoch {epoch + 1}")
                    break

            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        print(f"\n Training complete. Best Val Loss: {best_val_loss:.4f}")
    def train(self, epoch = 0):
        """
        Train the model
        :param epoch: How many epochs to train the model
        :return:
        """
        wafer_names = self.df_train['WaferName'].unique()
        train_names, val_names = train_test_split(wafer_names, test_size=0.05, random_state=42)

        train_dataset = WaferDataset(self.df_train, train_names)
        val_dataset = WaferDataset(self.df_train, val_names)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4, pin_memory=True)
        print('Start training')

        self.__train_model(self.model, train_loader, val_loader, self.optimizer, self.criterion, self.device, num_epochs=epoch)
    def LoadModel(self,save_path = os.path.join('Model','SaveModel', "best_model_new.pth")):
        self.model.load_state_dict(torch.load(save_path))

    def filter_high_yield_wafers(self,df, yield_threshold=0.7):
        """
        Split dataset into high-yield wafers for scratch detection, and low-yield wafers to ignore.

        Args:
            df (DataFrame): Full wafer data
            yield_threshold (float): Threshold for yield

        Returns:
            high_yield_df (DataFrame): Only high-yield wafers
            low_yield_wafer_list (list): List of low-yield wafer names
        """
        wafer_yields = []

        for wafer_name, wafer_df in df.groupby('WaferName'):
            total_dies = len(wafer_df)
            good_dies = wafer_df['IsGoodDie'].sum()
            yield_ratio = good_dies / total_dies
            wafer_yields.append((wafer_name, yield_ratio))

        high_yield_wafer_names = [name for name, y in wafer_yields if y >= yield_threshold]
        low_yield_wafer_names = [name for name, y in wafer_yields if y < yield_threshold]

        high_yield_df = df[df['WaferName'].isin(high_yield_wafer_names)]

        print(f" High-yield wafers: {len(high_yield_wafer_names)}")
        print(f" Low-yield wafers (ignored): {len(low_yield_wafer_names)}")

        return high_yield_df, low_yield_wafer_names

    def predict_wafer_df(self,model, test_df, cache_dir, device="cuda", save_path="wafers_test_with_predictions_fixed.csv"):
        """
        Predict scratch dies for each wafer and return updated DataFrame with 'IsScratchDie' column.

        Args:
            model: trained PyTorch model (ImprovedUNet).
            test_df: original test dataframe (must contain DieX, DieY, WaferName).
            cache_dir: path to directory with cached fixed .pt images.
            device: 'cuda' or 'cpu'.
            save_path: path to save the final CSV (optional).

        Returns:
            all_prediction: a list represent 'IsScratchDie' column (0 or 1).
        """
        model.eval()
        model.to(device)

        new_test_df = test_df.copy()
        all_predictions = []

        wafer_names = test_df['WaferName'].unique()

        for wafer_name in tqdm(wafer_names, desc="Predicting wafers"):
            # Load image from fixed coordinate cache
            img_path = os.path.join(cache_dir, "images", f"{wafer_name}.pt")
            image = torch.load(img_path).unsqueeze(0).to(device)  # shape: [1, 1, 256, 256]

            with torch.no_grad():
                output = model(image)
                prediction = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()  # shape: [256, 256]

            # Map prediction back to dies in test_df
            wafer_df = test_df[test_df['WaferName'] == wafer_name].copy()

            min_x, max_x = wafer_df['DieX'].min(), wafer_df['DieX'].max()
            min_y, max_y = wafer_df['DieY'].min(), wafer_df['DieY'].max()

            for _, row in wafer_df.iterrows():
                x = int(row['DieX'] - min_x)
                y = int(row['DieY'] - min_y)

                if 0 <= x < 256 and 0 <= y < 256:
                    scratch_pred = int(prediction[y, x])  # access pixel
                else:
                    scratch_pred = 0  # default fallback

                all_predictions.append(scratch_pred)

        # Add predictions to dataframe
        #new_test_df['IsScratchDie'] = all_predictions
        #new_test_df.to_csv(save_path, index=Fals

        return all_predictions
    def predict(self,df_frame = None, yield_threshold = 0.7,train_predict = False):
        """
        Predict scratch dies for each wafer and return updated DataFrame with 'IsScratchDie' column.

        Args:
            df_frame: original test dataframe (must contain DieX, DieY, WaferName).
            yield_threshold: Yield threshold for yield

        Returns:
            prediction: a list represent 'IsScratchDie' column (0 or 1).
        """
        df_use = df_frame if not train_predict else self.df_train
        df_test, bad_wafers = self.filter_high_yield_wafers(df_use,yield_threshold)
        wafer_names = df_test['WaferName'].unique()
        cache_dir = os.path.join('datamap', 'data','test') if not train_predict else os.path.join('datamap', 'data','cache')
        self.__build_wafer_cache(df_test, wafer_names, cache_dir=cache_dir,is_test = True)
        prediction = self.predict_wafer_df(self.model, df_test, cache_dir=cache_dir, device=self.device)
        return prediction

    def evaluate_die_level(self,true_labels, pred_labels,save_path="die_level_metrics.csv"):
        """
        true_labels: 1D numpy array of true IsScratchDie per die
        pred_labels: 1D numpy array of predicted IsScratchDie per die
        """

        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        accuracy = accuracy_score(true_labels, pred_labels)

        print(f"Die-level Evaluation Metrics:")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        # Save to CSV
        metrics_df = pd.DataFrame([{
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4)
        }])

        metrics_df.to_csv(save_path, index=False)
        print(f"Metrics saved to: {save_path}")









