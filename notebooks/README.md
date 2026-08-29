# Notebooks

The research side of Kiku: the creation of the multiclass neural network,
end to end. The Rust crate is the production runtime and does not depend on
anything here.

The network is multitask by construction. One decoder vocabulary carries,
beyond the text tokens, a class for every language, the task classes
(transcribe and translate), the no speech class, and a timestamp class every
20 ms. One softmax over that vocabulary makes every prediction, so a single
model performs language identification, voice activity detection,
segmentation, transcription, and translation into English.

Read them in this order:

1. `data_preparation.ipynb` builds the manifest with the language, task, and
   translation fields the multitask targets need.
2. `audio_frontend.ipynb` computes the 80 channel log Mel input and the
   training augmentation.
3. `tokenizer_training.ipynb` builds the BPE vocabulary and the multitask
   token map, and encodes full target sequences.
4. `model_architecture.ipynb` defines the encoder decoder network with the
   tied multiclass output layer.
5. `training.ipynb` trains all the tasks jointly in mixed batches.
6. `decoding.ipynb` runs inference over the multitask grammar with the
   reliability gates.
7. `evaluation.ipynb` measures per task and per language.
8. `export_checkpoint.ipynb` writes safetensors the Rust runtime loads.
