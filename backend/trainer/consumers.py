import json
import asyncio
import numpy as np

from channels.generic.websocket import AsyncWebsocketConsumer

from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from channels.db import database_sync_to_async
from .models import CustomDataset
import pandas as pd


# 📊 Dataset factory
def get_dataset(name):
    if name == "moons":
        return make_moons(n_samples=200, noise=0.2)
    elif name == "circles":
        return make_circles(n_samples=200, noise=0.1)
    elif name == "blobs":
        return make_blobs(n_samples=200, centers=2)
    else:
        return make_moons(n_samples=200, noise=0.2)

@database_sync_to_async
def get_custom_dataset(name):
    try:
        ds = CustomDataset.objects.filter(name=name).first()
        if not ds: return None, None
        df = pd.read_csv(ds.csv_file.path)

        target_col = None
        for col in ['target', 'label', 'y']:
            if col in [c.lower() for c in df.columns]:
                target_col = df.columns[[c.lower() for c in df.columns].index(col)]
                break
        if not target_col:
            target_col = df.columns[-1]
        
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        # Normalize Data to prevent visual scaling destruction
        import numpy as np
        if X.std(axis=0).all() != 0:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        return X, y
    except Exception as e:
        return None, None

# 🤖 Model factory
def get_model(name):
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(10,), warm_start=True, max_iter=1)

    elif name == "svm":
        return SVC(probability=True)

    elif name == "rf":
        return RandomForestClassifier(n_estimators=1, warm_start=True)

    elif name == "logreg":
        return LogisticRegression()

    elif name == "knn":
        return KNeighborsClassifier(n_neighbors=5)

    else:
        return MLPClassifier(hidden_layer_sizes=(10,), warm_start=True, max_iter=1)


# 🧠 Decision boundary
def compute_boundary(model, X):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(grid)[:, 1] # Probability of Class 1
    else:
        preds = model.predict(grid)

    return preds.reshape(xx.shape).tolist()


class TrainConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.model_id = self.scope["url_route"]["kwargs"]["model_id"]
        print(f"Connected: Model {self.model_id}")
        self.train_task = None
        self.X = None
        self.y = None
        self.model = None
        self.dataset_name = "moons"
        self.model_name = "mlp"
        await self.accept()

    async def disconnect(self, close_code):
        print(f"Disconnected: Model {self.model_id}")
        if self.train_task:
            self.train_task.cancel()

    async def receive(self, text_data):
        data = json.loads(text_data)
        action = data.get("action", "train")

        if action == "train":
            epochs = data.get("epochs", 100)
            self.dataset_name = data.get("dataset", "moons")
            self.model_name = data.get("model", "mlp")

            # 📊 Dataset
            if self.dataset_name in ["moons", "circles", "blobs"]:
                X, y = get_dataset(self.dataset_name)
            else:
                X, y = await get_custom_dataset(self.dataset_name)
                if X is None:
                    X, y = get_dataset("moons")
            self.X = X
            self.y = y

            # 🤖 Model
            self.model = get_model(self.model_name)

            if self.train_task:
                self.train_task.cancel()
            
            self.train_task = asyncio.create_task(self._run_training_loop(epochs))
            
        elif action == "update_point":
            idx = data.get("index")
            new_coords = data.get("new_coords")
            if self.X is not None and idx is not None and new_coords is not None:
                if 0 <= idx < len(self.X):
                    self.X[idx] = [new_coords[0], new_coords[1]]
                    if self.train_task:
                        self.train_task.cancel()
                    self.train_task = asyncio.create_task(self._run_training_loop(1, is_update=True))

    async def _run_training_loop(self, epochs, is_update=False):
        try:
            for epoch in range(epochs):
                if self.model_name == "mlp":
                    self.model.fit(self.X, self.y)
                    loss = self.model.loss_
                elif self.model_name == "rf":
                    if not is_update:
                        self.model.n_estimators = epoch + 1
                    self.model.fit(self.X, self.y)
                    loss = self.model.score(self.X, self.y)
                else:
                    if epoch == 0:
                        self.model.fit(self.X, self.y)
                    else:
                        break
                    loss = self.model.score(self.X, self.y)

                real_total_epochs = epochs if self.model_name in ["mlp", "rf"] and not is_update else 1

                boundary = compute_boundary(self.model, self.X)
                accuracy = self.model.score(self.X, self.y)

                metadata = {}
                if self.model_name == "svm":
                    if hasattr(self.model, "support_vectors_"):
                        metadata["support_vectors"] = self.model.support_vectors_.tolist()
                elif self.model_name == "logreg":
                    if hasattr(self.model, "coef_") and hasattr(self.model, "intercept_"):
                        metadata["weights"] = self.model.coef_[0].tolist()
                        metadata["bias"] = self.model.intercept_[0]

                x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
                y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5

                if not is_update:
                    await asyncio.sleep(0.15)

                await self.send(text_data=json.dumps({
                    "action": "update" if is_update else "train",
                    "epoch": epoch,
                    "total_epochs": real_total_epochs,
                    "loss": float(loss),
                    "boundary": boundary,
                    "accuracy": accuracy,
                    "predictions": self.model.predict(self.X).tolist() if hasattr(self.model, "predict") else [],
                    "points": self.X.tolist(),
                    "labels": self.y.tolist(),
                    "metadata": metadata,
                    "range": {"xMin": x_min, "xMax": x_max, "yMin": y_min, "yMax": y_max}
                }))
        except asyncio.CancelledError:
            pass