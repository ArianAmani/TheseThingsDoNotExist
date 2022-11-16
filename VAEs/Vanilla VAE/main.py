import click
from model import VanillaVAE
from data import create_dataset
import tensorflow as tf
from train import train
from sample import sample
from conf import batch_size, image_size, latent_dim
import os


@click.command()
@click.option('--train', 'to_train', is_flag=True)
@click.option('--epochs', default=20)
@click.option('--learning_rate', default=0.0005)
@click.option('--device', default='cpu')
@click.option('--save_dir', default='checkpoint')
@click.option('--save_name', default='vae')
@click.option('--sample', 'to_sample', is_flag=True)
@click.option('--num_samples', default=10)
@click.option('--output_dir', default='output')
@click.option('--weights_path', default='checkpoint/vae_', help="path to encoder and decoder weights without 'encoder' 'decoder' extensions")
def main(to_train, epochs, learning_rate, device, save_dir, save_name, to_sample, num_samples, output_dir, weights_path):

    if to_train:
        print('Training VAE:')
        print('Loading dataset...')
        train_dataset, val_dataset = create_dataset(
            batch_size=batch_size, img_size=image_size, download=True)

        print('Creating model...')
        vae = VanillaVAE(image_shape=(
            image_size, image_size, 3), latent_dim=latent_dim)

        print('Training model...')
        vae = train(vae, train_dataset, val_dataset, epochs=epochs,
                    learning_rate=learning_rate, device=device, save_dir=save_dir, save_name=save_name)

        print('Done!')

    if to_sample:
        print('Generating new images from VAE:')
        print('Creating model...')
        vae = VanillaVAE(image_shape=(
            image_size, image_size, 3), latent_dim=latent_dim)
        
        print('Loading weights...')
        vae.encoder.load_weights(weights_path + 'encoder/variables/variables')
        vae.decoder.load_weights(weights_path + 'decoder/variables/variables')

        print('Generating samples...')
        # (num_samples, img_size, img_size, 3) pixels in range [0, 1]
        samples = sample(vae, num_samples=num_samples, device=device)

        print('Saving generated images...')
        os.makedirs(output_dir, exist_ok=True)
        for i, image in enumerate(samples):
            image = (image.numpy() * 255).astype("int32")
            tf.keras.preprocessing.image.save_img(
                os.path.join(output_dir, f'sample_{i}.png'), image)

        print('Images saved in ' + output_dir)
        print('Done!')


if __name__ == '__main__':
    main()