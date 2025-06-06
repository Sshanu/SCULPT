import numpy as np
from sentence_transformers import SentenceTransformer
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

class Apex:
    def __init__(self, alpha=1.0, lambda_=1.0):
        """
        Initialize the Ridge Regression UCB model.

        :param alpha: Confidence bound parameter (exploration-exploitation trade-off)
        :param lambda_: Regularization parameter for ridge regression
        """
        self.alpha = alpha
        self.lambda_ = lambda_

        # Initialize Sentence Transformer model to get encoding dimensions
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.n_features = self.model.get_sentence_embedding_dimension()  # Set n_features based on encoding dimensions

        # Initialize the covariance matrix (X.T @ X + lambda * I)
        self.V = lambda_ * np.eye(self.n_features)
        
        # Initialize the X.T @ y vector
        self.V_inv = np.linalg.inv(self.V)  # To keep track of (X.T @ X + lambda * I)^-1
        self.b = np.zeros(self.n_features)  # This will hold X.T @ y
        
        # History of encoded sentences and corresponding rewards
        self.history_encoded_X = []  # Encoded context vectors
        self.history_y = []          # Rewards
        self.encoded_history = {}    # Dictionary to store encoded sentences

    def encode(self, sentences):
        # Check if the sentences have already been encoded
        encoded_sentences = []
        for sentence in sentences:
            if sentence not in self.encoded_history:
                encoded = self.model.encode([sentence])
                self.encoded_history[sentence] = encoded
            encoded_sentences.append(self.encoded_history[sentence])
        
        return np.vstack(encoded_sentences)  # Combine the encoded sentences into a single array

    def update_history(self, sentences, rewards):
        # Encode the new sentences and update history
        encoded_X = self.encode(sentences)
        self.history_encoded_X.extend(encoded_X)  # Store encoded sentences
        self.history_y.extend(rewards)             # Store corresponding rewards
        
        # Fit the model with the new encoded data
        self.fit(encoded_X, rewards)

    def fit(self, X, y):
        for x, reward in zip(X, y):
            x = x.reshape(-1, 1)  # Convert to column vector
            # Update V and b (X.T @ X and X.T @ y)
            self.V += x @ x.T
            self.b += reward * x.squeeze()

        # Update V inverse (to avoid computing inverse repeatedly)
        self.V_inv = np.linalg.inv(self.V)

    def predict(self, x):
        return x.T @ (self.V_inv @ self.b)

    def upper_confidence_bound(self, x):
        x = x.reshape(-1, 1)
        mean = self.predict(x)
        # Compute the exploration term (sqrt(x.T @ V_inv @ x))
        exploration = np.sqrt(x.T @ self.V_inv @ x)
        return mean + self.alpha * exploration

    def select_action(self, sentences):
        encoded_X = self.encode(sentences)
        ucb_values = [self.upper_confidence_bound(x) for x in encoded_X]
        best_action = np.argmax(ucb_values)
        return best_action
