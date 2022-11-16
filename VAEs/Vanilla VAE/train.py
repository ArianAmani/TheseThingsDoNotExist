import tensorflow as tf
import os


def train(vae, train_dataset, val_dataset, epochs=20, learning_rate=0.0005, device='cpu', save_dir='checkpoint', save_name='vae'):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    vae.compile(optimizer)
    
    
    if device == 'gpu':
        with tf.device('/device:GPU:0'):
            vae.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

    elif device == 'cpu':
        vae.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

    os.makedirs(save_dir, exist_ok=True)
    chk_path = os.path.join(save_dir, save_name)
    vae.encoder.save(chk_path + '_encoder')
    vae.decoder.save(chk_path + '_decoder')

    return vae