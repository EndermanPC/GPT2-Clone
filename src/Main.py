import torch
from matplotlib import pyplot as plt

from LanguageModel import LanguageModel
from Wraps import AutoregressiveWrapper
from Defent import create_training_sequences, tokenize_and_pad_training_data
from Model import Trainer, Generator
from Tokenizer import Tokenizer, get_device

class Runner(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def run(self):
        tokenizer = Tokenizer()

        embedding_dimension = 256
        max_sequence_length = 20
        number_of_tokens = tokenizer.size()

        model = AutoregressiveWrapper(LanguageModel(
            embedding_dimension=embedding_dimension,
            number_of_tokens=number_of_tokens,
            number_of_heads=4,
            number_of_layers=3,
            dropout_rate=0.1,
            max_sequence_length=max_sequence_length
        )).to(get_device())

        training_data = '. '.join([
            'beginoftext cats rule the world endoftext',
            'beginoftext dogs are the best endoftext',
            'beginoftext elephants have long trunks endoftext',
            'beginoftext monkeys like bananas endoftext',
            'beginoftext pandas eat bamboo endoftext',
            'beginoftext tigers are dangerous endoftext',
            'beginoftext zebras have stripes endoftext',
            'beginoftext lions are the kings of the savannah endoftext',
            'beginoftext giraffes have long necks endoftext',
            'beginoftext hippos are big and scary endoftext',
            'beginoftext rhinos have horns endoftext',
            'beginoftext penguins live in the arctic endoftext',
            'beginoftext polar bears are white endoftext'
        ])

        tokenized_and_padded_training_data = tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data)
        sequences = create_training_sequences(max_sequence_length, tokenized_and_padded_training_data)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        trainer = Trainer(model, tokenizer, optimizer)
        loss_per_epoch = trainer.train(sequences, epochs=120, batch_size=16)

        plt.plot(loss_per_epoch)
        plt.yscale('log')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        model.save_checkpoint('./Moloom')

        max_tokens_to_generate = 400
        generator = Generator(model, tokenizer)
        generated_text = generator.generate(
            max_tokens_to_generate=max_tokens_to_generate,
            prompt="beginoftext monkeys",
            padding_token=tokenizer.character_to_token('<pad>')
        )
        print(generated_text.replace('<pad>', ''))

Runner().run()