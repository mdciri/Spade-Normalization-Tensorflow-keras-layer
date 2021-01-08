import tensorflow as tf

class SpadeNorm(tf.keras.layers.Layer):

    def __init__(self, segmap, Nh=128, ks=3):
        super(SpadeNorm, self).__init__() 
        self.segmap = segmap # segmentation map
        self.Nh = Nh         # number of hidden filters
        self.ks = ks         # kernel size

    # Create the state of the layer (weights)
    def build(self, input_shape):  
        
        # number of channels in the inputs and in the segmentation map
        channels_seg = self.segmap.shape[-1]
        channels_inputs = input_shape[-1]
        
        # initialization BatchNorm weights
        self.gamma_BN = self.add_weight(shape=(input_shape[0],),
                                        initializer='ones', trainable=True)
        self.beta_BN = self.add_weight(shape=(input_shape[0],),
                                        initializer='zeros', trainable=True)
        
        # initializate kernels for the convolutions      
        self.kernel_alpha = self.add_weight(shape=(self.ks, self.ks, channels_seg, self.Nh), 
                                            initializer='random_normal', trainable=True)
        self.kernel_gamma = self.add_weight(shape=(self.ks, self.ks, self.Nh, channels_inputs), 
                                            initializer='random_normal', trainable=True)
        self.kernel_beta = self.add_weight(shape=(self.ks, self.ks, self.Nh, channels_inputs), 
                                           initializer='random_normal', trainable=True)

    # Defines the computation from inputs and labels to outputs
    def call(self, inputs):
        
        # normalize the inputs
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        norm_inputs = tf.nn.batch_normalization(inputs, mean, variance, 
                                                offset=self.beta_BN, scale=self.gamma_BN,
                                                variance_epsilon=1e-5)

        # resize the segmap
        _, inputs_height, inputs_width, _ = inputs.shape
        segmap_resized = tf.image.resize(self.segmap, size=[inputs_height, inputs_width], method='nearest') 

        # first convolution with activation
        actv = tf.nn.relu(tf.nn.conv2d(segmap_resized, self.kernel_alpha, strides=[1,1,1,1], padding='SAME'))

        # calculate gamma and beta
        beta = tf.nn.conv2d(actv, self.kernel_beta, strides=[1,1,1,1], padding='SAME')
        gamma = tf.nn.conv2d(actv, self.kernel_gamma, strides=[1,1,1,1], padding='SAME')  

        # outputs
        outputs = norm_inputs * (1 + gamma) + beta       
        
        return outputs
