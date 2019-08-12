from .base_graph import BaseGraph
import itertools
import tensorflow as tf
import tensorflow.keras.layers as layers
from robonet.video_prediction.flow_ops import image_warp
from robonet.video_prediction.layers.dnaflow_rnn_cell import apply_cdna_kernels, RELU_SHIFT
from robonet.video_prediction.models.lstm_modules import GaussianLSTM


def _cast_down(tensor, hparams):
    if hparams.float16 and hparams.use_tpu:
        return tf.cast(tensor, tf.bfloat16)
    elif hparams.float16:
        return tf.cast(tensor, tf.float16)
    return tensor


def _cast_up(tensor):
    if tensor.dtype != tf.float32:
        return tf.cast(tensor, tf.float32)
    return tensor


class SVG_VGGConvGraph(BaseGraph):
    def build_graph(self, mode, inputs, hparams, scope_name='flow_generator'):
        self._scope_name = scope_name
        #TODO "implement state conditioning"
        assert not hparams.use_states

        with tf.variable_scope(self._scope_name) as graph_scope:
            self._init_layers(hparams, inputs, mode)
            T, B, H, W, C = inputs['images'].get_shape().as_list()

            enc_lstm_state, dec_lstm_state = None, None
            previous_encs = []

            previous_latents = []

            outputs = {}

            for t in range(hparams.sequence_length - 1):
                norm_ctr = 0
                print('building graph for t={}'.format(t), end="\r")
                action_state_vector = _cast_down(inputs['actions'][t], hparams)
                if t < hparams.context_frames:
                    input_image = _cast_down(inputs['images'][t], hparams)
                else:
                    casted_real = _cast_down(inputs['images'][t], hparams)
                    casted_gen = _cast_down(outputs['gen_images'][-1], hparams)
                    input_image = tf.where(self._ground_truth[t], casted_real, casted_gen)

                gt_next_image = _cast_down(inputs['images'][t+1], hparams)

                # encoder convs
                encoded_imgs = input_image
                encoded_gt = gt_next_image
                for i, op in enumerate(self._enc_ops):
                    encoded_imgs = op(encoded_imgs)
                    encoded_gt = op(encoded_gt)
                    if isinstance(op, layers.Conv2D):
                        encoded_imgs = _cast_up(encoded_imgs)
                        encoded_gt = _cast_up(encoded_gt)
                        encoded_imgs = tf.contrib.layers.instance_norm(encoded_imgs, activation_fn=tf.nn.relu, 
                                                                        scope='norm{}'.format(norm_ctr), reuse = t > 0)
                        norm_ctr += 1
                        encoded_gt = tf.contrib.layers.instance_norm(encoded_gt, activation_fn=tf.nn.relu,
                                                                        scope='norm{}'.format(norm_ctr), reuse = t > 0)
                        encoded_imgs = _cast_down(encoded_imgs, hparams)
                        encoded_gt = _cast_down(encoded_gt, hparams)
                        norm_ctr += 1

                # encode actions and append to hidden state
                enc_append = tf.reshape(self._enc_append(action_state_vector), (B, int(H // 8), int(W // 8), 
                                        hparams.action_append_channels))
                encoded_imgs = self._enc_conv(tf.concat((encoded_imgs, enc_append), -1))
                encoded_imgs = tf.contrib.layers.instance_norm(_cast_up(encoded_imgs), activation_fn=tf.nn.relu, 
                                                                scope='norm{}'.format(norm_ctr), reuse = t > 0)
                norm_ctr += 1

                # put encoded images through inference model and learned prior LSTMs
                if t == 0:
                    self.posterior_lstm.init_hidden()
                    self.learned_prior_lstm.init_hidden()
                z_post, mu_post, logvar_post = self.posterior_lstm(encoded_gt)
                z_prior, mu_prior, logvar_prior = self.learned_prior_lstm(encoded_imgs)

                previous_latents.append({
                    'prior': {
                        'mu': mu_prior,
                        'logvar': logvar_prior
                     },
                    'post': {
                        'mu': mu_post,
                        'logvar': logvar_post
                    }
                })

                # append z_t to hidden state
                enc_append_z = tf.reshape(self._enc_append_z(z_post), (B, int(H // 8), int(W // 8),
                                                                       hparams.z_append_channels))
                encoded_imgs = self._enc_conv(tf.concat((encoded_imgs, enc_append_z), -1))
                encoded_imgs = tf.contrib.layers.instance_norm(_cast_up(encoded_imgs), activation_fn=tf.nn.relu,
                                                               scope='norm{}'.format(norm_ctr), reuse=t > 0)
                norm_ctr += 1

                # encoder lstm cell
                if t == 0:
                    enc_lstm_state = self._enc_lstm.get_initial_state(encoded_imgs[:, None])
                enc_out, enc_lstm_state = self._enc_lstm.cell(encoded_imgs, enc_lstm_state)

                # decoder attention
                enc_out = _cast_down(enc_out, hparams)
                flatten_enc = tf.reshape(enc_out, (B, -1))
                if t == 0:
                    previous_encs = [flatten_enc]
                    attention_enc = enc_out
                else:
                    previous_encs.append(flatten_enc)
                    dot_prods = [tf.reduce_sum(flatten_enc * x, 1, keepdims=True) for x in previous_encs]
                    attention_weights = tf.nn.softmax(tf.concat(dot_prods, axis=1), axis=1)
                    attention_enc = tf.reduce_sum(attention_weights[:, :, None] * tf.concat([p[:, None] for p in previous_encs], 1), 1)
                    attention_enc = tf.reshape(attention_enc, [B, int(H // 8), int(W // 8), hparams.lstm_filters])
                attention_enc = _cast_up(attention_enc)

                # decoder lstm cell
                if t == 0:
                    dec_lstm_state = self._dec_lstm.get_initial_state(attention_enc[:, None])
                dec_lstm_out, dec_lstm_state = self._dec_lstm.cell(attention_enc, dec_lstm_state)
                
                # decoder convs
                if t < hparams.context_frames - 1:   # no frame predictions for extra context frames
                    continue
                
                decoder_out = _cast_down(dec_lstm_out, hparams)
                for op in self._dec_ops:
                    decoder_out = op(decoder_out)
                    if isinstance(op, layers.Conv2D):
                        decoder_out = _cast_up(decoder_out)
                        decoder_out = tf.contrib.layers.instance_norm(decoder_out, activation_fn=tf.nn.relu, 
                                                                        scope='norm{}'.format(norm_ctr), reuse = t >= hparams.context_frames)
                        decoder_out = _cast_down(decoder_out, hparams)
                        norm_ctr += 1

                # predict flows
                if hparams.use_flows:
                    kernel_convs, mask_convs = tf.split(decoder_out, 2, axis=-1)
                    kernel_convs = tf.transpose(tf.reshape(kernel_convs, (B, -1, hparams.n_flows)), (0, 2, 1))
                    kernels = tf.nn.relu(self._kernel_top(kernel_convs - RELU_SHIFT)) + RELU_SHIFT
                    kernels = tf.transpose(kernels, (0, 2, 1))
                    kernels = tf.reshape(kernels / tf.reduce_sum(kernels, axis=1, keepdims=True), (B, hparams.cdna_kernel_size, -1, hparams.n_flows))
                    warped_images = tf.stack(apply_cdna_kernels(input_image, kernels), axis=-1)

                    masks = tf.expand_dims(tf.nn.softmax(self._mask_top(mask_convs)), axis=-2)

                    outputs['gen_images'] = outputs.get('gen_images', []) + [_cast_up(tf.reduce_sum(warped_images * masks, axis=-1))]
                else:
                    outputs['gen_images'] = outputs.get('gen_images', []) + [_cast_up(self._top(decoder_out))]

            outputs['gen_images'] = tf.concat([pred[None] for pred in outputs['gen_images']], 0)
            outputs['ground_truth_sampling_mean'] = tf.reduce_mean(tf.to_float(self._ground_truth[hparams.context_frames:]))
            outputs['enc_latents'] = previous_latents
        return outputs

    def _init_layers(self, hparams, inputs, mode):
        T, B, H, W, C = inputs['images'].get_shape().as_list()


        self._conv_1_1 = layers.Conv2D(hparams.enc_filters[0], hparams.kernel_size, padding='same')
        self._conv_1_2 = layers.Conv2D(hparams.enc_filters[0], hparams.kernel_size, padding='same')
        self._pool_1 = layers.MaxPool2D()

        self._conv_2_1 = layers.Conv2D(hparams.enc_filters[1], hparams.kernel_size, padding='same')
        self._conv_2_2 = layers.Conv2D(hparams.enc_filters[1], hparams.kernel_size, padding='same')
        self._conv_2_3 = layers.Conv2D(hparams.enc_filters[1], hparams.kernel_size, padding='same')
        self._pool_2 = layers.MaxPool2D()

        self._conv_3_1 = layers.Conv2D(hparams.enc_filters[2], hparams.kernel_size, padding='same')
        self._conv_3_2 = layers.Conv2D(hparams.enc_filters[2], hparams.kernel_size, padding='same')
        self._conv_3_3 = layers.Conv2D(hparams.enc_filters[2], hparams.kernel_size, padding='same')
        self._conv_3_4 = layers.Conv2D(hparams.enc_filters[2], hparams.kernel_size, padding='same')
        self._pool_3 = layers.MaxPool2D()

        self._enc_ops = [self._conv_1_1, self._conv_1_2, self._pool_1, self._conv_2_1, self._conv_2_2, self._conv_2_3,
                        self._pool_2, self._conv_3_1, self._conv_3_2, self._conv_3_3, self._conv_3_4, self._pool_3]

        enc_H, enc_W = [int(x // 8) for x in (H, W)]

        self.learned_prior_lstm = GaussianLSTM(inputs['images'].get_shape().as_list(), enc_H, enc_W,
                                               hparams.latent_dim, hparams.glstm_hidden_size, hparams.glstm_nlayers,
                                               hparams.glstm_kernel_size)

        self.posterior_lstm = GaussianLSTM(inputs['images'].get_shape().as_list(), enc_H, enc_W,
                                           hparams.latent_dim, hparams.glstm_hidden_size, hparams.glstm_nlayers,
                                           hparams.glstm_kernel_size)

        self._enc_append = layers.Dense(enc_H * enc_W * hparams.action_append_channels)
        self._enc_append_z = layers.Dense(enc_H * enc_W * hparams.z_append_channels)
        self._enc_conv = layers.Conv2D(hparams.lstm_filters, 1, padding='same')

        self._enc_lstm = layers.ConvLSTM2D(hparams.lstm_filters, hparams.kernel_size, padding = 'same')
        self._enc_lstm.cell.build([B, T, enc_H, enc_W, hparams.lstm_filters])
        self._dec_lstm = layers.ConvLSTM2D(hparams.lstm_filters, hparams.kernel_size, padding='same')
        self._dec_lstm.cell.build([B, T, enc_H, enc_W, hparams.lstm_filters])

        self._conv_tranpose_1 = layers.Conv2DTranspose(hparams.lstm_filters, 2, strides=2)
        self._dec_conv_1_1 = layers.Conv2D(hparams.dec_filters[0], hparams.kernel_size, padding='same')
        self._dec_conv_1_2 = layers.Conv2D(hparams.dec_filters[0], hparams.kernel_size, padding='same')
        self._dec_conv_1_3 = layers.Conv2D(hparams.dec_filters[0], hparams.kernel_size, padding='same')
        self._dec_conv_1_4 = layers.Conv2D(hparams.dec_filters[0], hparams.kernel_size, padding='same')

        self._conv_tranpose_2 = layers.Conv2DTranspose(hparams.dec_filters[0], 2, strides=2)
        self._dec_conv_2_1 = layers.Conv2D(hparams.dec_filters[1], hparams.kernel_size, padding='same')
        self._dec_conv_2_2 = layers.Conv2D(hparams.dec_filters[1], hparams.kernel_size, padding='same')
        self._dec_conv_2_3 = layers.Conv2D(hparams.dec_filters[1], hparams.kernel_size, padding='same')

        self._conv_tranpose_3 = layers.Conv2DTranspose(hparams.dec_filters[1], 2, strides=2)
        self._dec_ops = [self._conv_tranpose_1, self._dec_conv_1_1, self._dec_conv_1_2, self._dec_conv_1_3, self._dec_conv_1_4,
                        self._conv_tranpose_2, self._dec_conv_2_1, self._dec_conv_2_2, self._dec_conv_2_3,
                        self._conv_tranpose_3]
        
        if hparams.use_flows:
            self._kernel_mask_conv = layers.Conv2D(hparams.n_flows * 2, 1, padding='same')
            self._dec_ops.append(self._kernel_mask_conv)

            # kernel prediction
            self._kernel_top = layers.Dense(hparams.cdna_kernel_size ** 2)
            # mask prediction
            self._mask_top = layers.Conv2D(hparams.n_flows, hparams.kernel_size, padding='same')
        else:
            self._dec_conv_3_1 = layers.Conv2D(3, hparams.kernel_size, padding='same')
            self._top = layers.Conv2D(3, hparams.kernel_size, padding='same')

            self._dec_ops.append(self._dec_conv_3_1)

        ground_truth_sampling_shape = [hparams.sequence_length - 1 - hparams.context_frames, B]
        if hparams.schedule_sampling == 'none' or mode != tf.estimator.ModeKeys.TRAIN:
            ground_truth_sampling = tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape)
        elif hparams.schedule_sampling in ('inverse_sigmoid', 'linear'):
            if hparams.schedule_sampling == 'inverse_sigmoid':
                k = hparams.schedule_sampling_k
                start_step = hparams.schedule_sampling_steps[0]
                iter_num = tf.to_float(tf.train.get_or_create_global_step())
                prob = (k / (k + tf.exp((iter_num - start_step) / k)))
                prob = tf.cond(tf.less(iter_num, start_step), lambda: 1.0, lambda: prob)
            elif hparams.schedule_sampling == 'linear':
                start_step, end_step = hparams.schedule_sampling_steps
                step = tf.clip_by_value(tf.train.get_or_create_global_step(), start_step, end_step)
                prob = 1.0 - tf.to_float(step - start_step) / tf.to_float(end_step - start_step)
            log_probs = tf.log([1 - prob, prob])
            ground_truth_sampling = tf.multinomial([log_probs] * B, ground_truth_sampling_shape[0])
            ground_truth_sampling = tf.cast(tf.transpose(ground_truth_sampling, [1, 0]), dtype=tf.bool)
            # Ensure that eventually, the model is deterministically
            # autoregressive (as opposed to autoregressive with very high probability).
            ground_truth_sampling = tf.cond(tf.less(prob, 0.001),
                                            lambda: tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape),
                                            lambda: ground_truth_sampling)
        else:
            raise NotImplementedError
        ground_truth_context = tf.constant(True, dtype=tf.bool, shape=[hparams.context_frames, B])
        self._ground_truth = tf.concat([ground_truth_context, ground_truth_sampling], axis=0)

    @property
    def vars(self):
        return tf.trainable_variables(self._scope_name)

    @staticmethod
    def default_hparams():
        default_params =  {
            "enc_filters": [32, 128, 256],
            "lstm_filters": 256,
            "dec_filters": [128, 64],
            "kernel_size": 3,
            'action_append_channels': 2,
            'z_append_channels': 2,
            'latent_dim': 8,
            'glstm_hidden_size': 32,
            'glstm_n_layers': 1,
            'glstm_kernel_size': 3,

            "use_flows": True,
            "n_flows": 12,
            "cdna_kernel_size": 10,

            'schedule_sampling': "inverse_sigmoid",
            'schedule_sampling_k': 900.0,
            'schedule_sampling_steps': [0, 100000],

            "float16": False                        # float 16 is very unstable at the moment
        }
        return dict(itertools.chain(BaseGraph.default_hparams().items(), default_params.items()))
