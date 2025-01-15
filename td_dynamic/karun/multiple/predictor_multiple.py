import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

class OracleActionState:
    def __init__(self, action_state):
        self.action_state = action_state

    def transform(self, X):
        unique_rows, unique_indices = np.unique(self.action_state, axis=0, return_inverse=True)
        new_column = unique_indices.reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        dummy_variables = encoder.fit_transform(new_column)
        dummy_variables = pl.DataFrame(dummy_variables).rename(
            {f"column_{i}": f"combination_{i}" for i in range(dummy_variables.shape[1])}
        )
        dummy_variables = pl.concat([pl.DataFrame(unique_rows), dummy_variables], how="horizontal")
        X = pl.DataFrame(X)
        X = X.join(
            dummy_variables,
            on=[col for col in X.columns if col.startswith("column_")],
            how="left",
        )
        X = X.select(*[col for col in X.columns if col.startswith("combination_")])
        X = np.array(X)
        return X

class PredictorBase:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def score(self, *args, **kwargs):
        raise NotImplementedError


class LinearRegressionPredictor(PredictorBase):
    def __init__(
        self,
        equilibrium,
        predictor_type="polynomial",
        degree=2,
        fit_intercept=False,
        copy_X=True,
        n_jobs=None,
        positive=False,
        i=0,
    ):
        super().__init__()
        self.basis = self.fit_basis_action_state(
            equilibrium=equilibrium, predictor_type=predictor_type, degree=degree, i=i
        )
        self.model = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)

    def fit_basis_action_state(self, equilibrium, predictor_type, degree, i):
        action_state = equilibrium.action_state["action_state_value"]
        num_firm = equilibrium.num["firm"]
        if predictor_type == "polynomial":
            basis = PolynomialFeatures(degree=degree)
            basis = basis.fit(np.unique(action_state[:, [i, *range(num_firm, action_state.shape[1])]], axis=0))
            return basis
        elif predictor_type == "oracle":
            basis = OracleActionState(action_state=np.unique(action_state[:, [i, *range(num_firm, action_state.shape[1])]], axis=0))
            return basis
        else:
            raise ValueError("Invalid basis predictor_type")

    def fit(self, X, y):
        X = self.basis.transform(X)
        return self.model.fit(X, y)

    def predict(self, X):
        X = self.basis.transform(X)
        return self.model.predict(X)

    def score(self, X, y):
        X = self.basis.transform(X)
        return self.model.score(X, y)


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MLPPredictor(PredictorBase):
    def __init__(
        self,
        equilibrium,
        predictor_type="polynomial",
        degree=2,
        hidden_layer_sizes=(100,),
        learning_rate=0.001,
        batch_size=32,
        num_epochs=200,
        device="cuda" if torch.cuda.is_available() else "cpu",
        i=0,
    ):
        super().__init__()
        self.basis = self.fit_basis_action_state(
            equilibrium=equilibrium, predictor_type=predictor_type, degree=degree, i=i
        )
        self.scaler = StandardScaler()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.model = None

    def fit_basis_action_state(self, equilibrium, predictor_type, degree, i):
        action_state = equilibrium.action_state["action_state_value"]
        num_firm = equilibrium.num["firm"]
        if predictor_type == "polynomial":
            basis = PolynomialFeatures(degree=degree)
            basis = basis.fit(np.unique(action_state[:, [i, *range(num_firm, action_state.shape[1])]], axis=0))
            return basis
        elif predictor_type == "oracle":
            basis = OracleActionState(action_state=np.unique(action_state[:, [i, *range(num_firm, action_state.shape[1])]], axis=0))
            return basis
        else:
            raise ValueError("Invalid basis predictor_type")

    def fit(self, X, y):
        X = self.basis.transform(X)
        X = self.scaler.fit_transform(X)
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)

        input_size = X.shape[1]
        output_size = y.shape[1] if len(y.shape) > 1 else 1

        self.model = MLPModel(input_size, self.hidden_layer_sizes, output_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

        return self

    def predict(self, X):
        X = self.basis.transform(X)
        X = self.scaler.transform(X)
        X = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            # Use DataLoader for batch prediction
            dataset = torch.utils.data.TensorDataset(X)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            predictions = []
            for batch in dataloader:
                batch_predictions = self.model(batch[0]).squeeze()
                predictions.append(batch_predictions)
            predictions = torch.cat(predictions, dim=0)
        return predictions.cpu().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

class OracleStateValue:
    def __init__(self, state_value):
        self.state_value = state_value

    def transform(self, X):
        unique_rows, unique_indices = np.unique(self.state_value, axis=0, return_inverse=True)
        new_column = unique_indices.reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        dummy_variables = encoder.fit_transform(new_column)
        dummy_variables = pl.DataFrame(dummy_variables).rename(
            {f"column_{i}": f"combination_{i}" for i in range(dummy_variables.shape[1])}
        )
        dummy_variables = pl.concat([pl.DataFrame(unique_rows), dummy_variables], how="horizontal")
        X = pl.DataFrame(X)
        X = X.join(
            dummy_variables,
            on=[col for col in X.columns if col.startswith("column_")],
            how="left",
        )
        X = X.select(*[col for col in X.columns if col.startswith("combination_")])
        X = np.array(X)
        return X

class CCPLogisticRegressionPredictor(PredictorBase):
    def __init__(
        self,
        equilibrium,
        ccp_predictor_type="polynomial",
        degree=2
    ):
        super().__init__()
        self.basis = self.fit_basis_state_value(
            equilibrium=equilibrium, ccp_predictor_type=ccp_predictor_type, degree=degree
        )
        self.model = None  # Placeholder for the fitted MNLogit model

    def fit_basis_state_value(self, equilibrium, ccp_predictor_type, degree):
        state_value = equilibrium.result.select(
            *[col for col in equilibrium.result.columns if col.startswith("state_value_")]
        )
        if ccp_predictor_type == "polynomial":
            basis = PolynomialFeatures(degree=degree)
            basis = basis.fit(np.unique(state_value, axis=0))
            return basis
        elif ccp_predictor_type == "oracle":
            basis = OracleStateValue(state_value=np.unique(state_value, axis=0))
            return basis
        else:
            raise ValueError("Invalid basis ccp_predictor_type")

    def fit(self, X, y):
        X_transformed = self.basis.transform(X)
        self.model = sm.MNLogit(y, sm.add_constant(X_transformed)).fit()
        return self

    def predict(self, X):
        X_transformed = self.basis.transform(X)
        return self.model.predict(sm.add_constant(X_transformed))

    def score(self, X, y):
        if self.model is None:
            raise ValueError("Model has not been fitted successfully.")
        y_pred = self.predict(X).argmax(axis=1)  # Use argmax to get the predicted class
        return (y_pred == y).mean()  # Calculate accuracy