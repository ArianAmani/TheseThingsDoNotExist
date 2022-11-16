# Vanilla Variational Autoencoder

## You can train or sample from a VAE:

 - example:
 ```
    Train: 
      python main.py --train --epochs=20 --learning_rate=0.0005 --device=gpu --save_dir=checkpoint --save_name=vae

    Sample:
      python main.py --sample --num_samples=10 --device=gpu --output_dir=output --weights_path=checkpoint/vae_
 ```

 
 ## Samples generated after just 20 epochs:
 (Weights in `checkpoint_20_epochs` directory)
 
 `python3 main.py --sample --num_samples=4 --output_dir=output --weights_path=checkpoint_20_epochs/vae_`
<img width="1604" src="/VAEs/Vanilla_VAE/output/sample_0.png?raw=true">  |  <img width="1604" src="/VAEs/Vanilla_VAE/output/sample_2.png?raw=true">
:-------------------------:|:-------------------------:
<img width="1604" src="/VAEs/Vanilla_VAE/output/sample_1.png?raw=true">  |  <img width="1604" src="/VAEs/Vanilla_VAE/output/sample_3.png?raw=true">
