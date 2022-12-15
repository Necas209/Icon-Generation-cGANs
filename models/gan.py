from keras.models import Model
from tensorflow.keras.optimizers import Adam


# define the combined generator and discriminator model, for updating the generator
def create_gan(g_model: Model, d_model: Model, lr: float, beta_1: float) -> Model:
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = Adam(learning_rate=lr, beta_1=beta_1)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt
    )
    return model
