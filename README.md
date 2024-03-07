# address_ASR

## Data Preparing
Beside ASR data, you have to prepare a list of reference bias audio to contruct a bias database for similarity audio search, in our setting we using a list of TTS audio generated from a list of Vietnamese address phrases.

To generate a bias database:
```bash
python address_database.py  --dim 768   # using wav2vec2.0 base
                            --num_chunk 1 
                            --data_path audio/bias_audio/ 
                            --save_path database/address_db.pt 
                            --model_path path\to\wav2vec2.0_model 
```
## Train model
To train the model you need modify the hyperparameter in the ./config folder

run training process by run

```bash
python train.py
```