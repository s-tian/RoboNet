import tensorflow as tf
import tensorflow.keras.layers as layers


class GaussianLSTM(layers.Layer):

    # Gaussian LSTM like from https://github.com/edenton/svg/blob/master/models/lstm.py

    def __init__(self, input_img_shape, enc_H, enc_W, output_size, hidden_size, n_layers, kernel_size):
        super().__init__()
        self.input_img_shape = input_img_shape
        self.T, self.B, self.W, self.H, self.C = self.input_img_shape
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self._mu_dense = layers.Dense(output_size)
        self._logvar_dense = layers.Dense(output_size)

        self._lstm_layers = []

        for _ in range(n_layers):
            lstm_layer = layers.ConvLSTM2D(hidden_size, kernel_size, padding='same')
            lstm_layer.cell.build([self.B, self.T, enc_H, enc_W, hidden_size])
            self._lstm_layers.append(lstm_layer)

        self.hidden = None

    def init_hidden(self, inputs):
        self.hidden = []
        for layer in self._lstm_layers:
            self.hidden.append(layer.get_initial_state(inputs))

    def __call__(self, inp):
        assert self.hidden is not None, 'Hidden layers have not been initialized'

        for i, layer in enumerate(self._lstm_layers):
            out, self.hidden[i] = layer.cell(inp, self.hidden[i])
            inp = out

        lstm_out_flat = tf.reshape(inp, (self.B, -1))
        mu = self._mu_dense(lstm_out_flat)
        logvar = self._logvar_dense(lstm_out_flat)

        sample = tf.random.normal(self.B, self.hidden_size) * tf.math.exp(0.5 * logvar) + mu

        return sample, mu, logvar






