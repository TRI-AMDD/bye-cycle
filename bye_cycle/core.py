import tensorflow as tf
from dataclasses import dataclass

@dataclass
class DegredationModelHyperParams:
    batch_size: int = 32
    lr: float = 1e-3
    hidden_dim: int = 32
    activation: str = 'tanh'
    input_window_size: int = 50
    reg_strength: float = 0.1
    drop_rate: float = 0.1
    interpolated_time_length: int = 100
    early_stopping_patience: int = 20

class DegredationModel(tf.keras.layers.Layer):
    
    def __init__(self, hyperparams: DegredationModelHyperParams, **kwargs):
        super(DegredationModel, self).__init__(**kwargs)
        self.batch_size = hyperparams.batch_size
        self.lr = hyperparams.lr
        self.hidden_dim = hyperparams.hidden_dim
        self.activation = hyperparams.activation
        self.input_window_size = hyperparams.input_window_size
        self.reg_strength = hyperparams.reg_strength  
        self.drop_rate = hyperparams.drop_rate
        self.interpolated_time_length = hyperparams.interpolated_time_length
        self.early_stopping_patience = hyperparams.early_stopping_patience
    
    def _build_keras_model(self):
        inputs = tf.keras.Input(shape=(self.input_window_size, self.interpolated_time_length,2))

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(self.hidden_dim, self.interpolated_time_length),
                                            input_shape=(self.input_window_size, self.interpolated_time_length, 2))(inputs)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(self.drop_rate))(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling1D())(x)

        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim, return_sequences=False))(x)

        x = tf.keras.layers.Dense(self.hidden_dim,
                                  kernel_regularizer=tf.keras.regularizers.L1(l1=self.reg_strength),
                                  activation=self.activation)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(self.drop_rate)(x)
        x = tf.keras.layers.Dense(self.hidden_dim//2,
                                  kernel_regularizer=tf.keras.regularizers.L1(l1=self.reg_strength),
                                  activation=self.activation)(x)

        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(self.drop_rate)(x)
        yhat = tf.keras.layers.Dense(2,
                                     kernel_regularizer=tf.keras.regularizers.L1(l1=self.reg_strength))(x) 
        model = tf.keras.Model(inputs=inputs, outputs=yhat)
        return model

    def compile(self):
        tf.keras.backend.clear_session()
        self.model = self._build_keras_model()
        opt = tf.optimizers.Adam(self.lr)
        self.model.compile(opt, loss='mae')
        return self.model
    
    def fit(self, X_train, y_train, validation_data, **kwargs):
        if "verbose" not in kwargs:
            kwargs["verbose"] = 0
        if "epochs" not in kwargs:
            kwargs["epochs"] = 50

        self.history = self.model.fit(X_train, y_train,
                            validation_data=validation_data, 
                            callbacks=[tf.keras.callbacks.ReduceLROnPlateau(
                                monitor="val_loss", factor=0.9, patience=5, min_lr=1e-5
                                ), tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                        restore_best_weights=True, patience=self.early_stopping_patience)],
                           batch_size=self.batch_size,  **kwargs)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X_test):
        return self.model.predict(X_test)        

    def summary(self):
        return self.model.summary()
    
    def history(self):
        return self.model.history()
        


