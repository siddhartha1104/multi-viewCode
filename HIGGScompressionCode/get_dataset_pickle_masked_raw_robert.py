import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm

# Updated imports for Gemma/Llama
from transformers import (
    GemmaTokenizer, GemmaForCausalLM,
    LlamaTokenizer, LlamaForCausalLM,
    BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig,
    BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, 
    RobertaTokenizer, RobertaForSequenceClassification
)

# HIGGS compression (assuming it's available)
try:
    from higgs import HIGGSCompressor
    HIGGS_AVAILABLE = True
except ImportError:
    print("[WARNING] HIGGS not available, using standard models")
    HIGGS_AVAILABLE = False

from data_masked_raw_robert import ZuCo_dataset
from model_decoding_spectro import BrainTranslator, BrainTranslatorNaive
from config import get_config


class CompressedBrainTranslator(nn.Module):
    """Brain Translator with HIGGS-compressed language model"""
    
    def __init__(self, model_name, compression_ratio=0.5, use_compression=True):
        super().__init__()
        self.model_name = model_name
        self.use_compression = use_compression and HIGGS_AVAILABLE
        
        # Load base language model
        if 'gemma' in model_name.lower():
            self.tokenizer = GemmaTokenizer.from_pretrained('google/gemma-2b-it')
            base_model = GemmaForCausalLM.from_pretrained('google/gemma-2b-it')
        elif 'llama' in model_name.lower():
            self.tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
            base_model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
        else:
            # Fallback to BART
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            base_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        
        # Apply HIGGS compression if available
        if self.use_compression:
            print(f"[INFO] Applying HIGGS compression with ratio {compression_ratio}")
            self.compressor = HIGGSCompressor(compression_ratio=compression_ratio)
            self.language_model = self.compressor.compress(base_model)
        else:
            self.language_model = base_model
        
        # Brain signal processing components (you'll need to adapt these)
        self.brain_encoder = self._build_brain_encoder()
        self.projection_layer = self._build_projection_layer()
    
    def _build_brain_encoder(self):
        """Build the EEG signal encoder - adapt based on your spectro model"""
        # This is a placeholder - replace with your actual brain encoder architecture
        return nn.Sequential(
            nn.Linear(1024, 512),  # Adjust input size based on your EEG features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )
    
    def _build_projection_layer(self):
        """Project brain embeddings to language model dimension"""
        lm_hidden_size = self.language_model.config.hidden_size
        return nn.Linear(256, lm_hidden_size)  # 256 from brain encoder output
    
    def forward(self, brain_signals, target_ids=None, attention_mask=None):
        # Encode brain signals
        brain_embeddings = self.brain_encoder(brain_signals)
        projected_embeddings = self.projection_layer(brain_embeddings)
        
        # Generate text using (possibly compressed) language model
        if hasattr(self.language_model, 'generate'):
            # For generation mode
            outputs = self.language_model(
                inputs_embeds=projected_embeddings,
                attention_mask=attention_mask,
                labels=target_ids
            )
        else:
            # Fallback
            outputs = self.language_model(projected_embeddings)
        
        return outputs


def setup_model_and_tokenizer(model_name, use_compression=True, compression_ratio=0.5):
    """Setup model and tokenizer based on configuration"""
    
    if model_name in ['BrainTranslatorGemma', 'BrainTranslatorLlama']:
        # Use our new compressed models
        model = CompressedBrainTranslator(
            model_name=model_name, 
            use_compression=use_compression,
            compression_ratio=compression_ratio
        )
        tokenizer = model.tokenizer
        
    elif model_name in ['BrainTranslator', 'BrainTranslatorNaive']:
        # Original BART-based models
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        if model_name == 'BrainTranslator':
            model = BrainTranslator()
        else:
            model = BrainTranslatorNaive()
            
    elif model_name == 'BertGeneration':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        config = BertConfig.from_pretrained("bert-base-cased")
        config.is_decoder = True
        model = BertLMHeadModel.from_pretrained("bert-base-cased", config=config)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model, tokenizer


if __name__ == '__main__':
    args = get_config('train_decoding')

    ''' config param'''
    dataset_setting = 'unique_sent'

    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']

    batch_size = args['batch_size']

    model_name = args['model_name']
    # New model options:
    # model_name = 'BrainTranslatorGemma'
    # model_name = 'BrainTranslatorLlama'
    
    # HIGGS compression settings
    use_compression = args.get('use_higgs_compression', True)
    compression_ratio = args.get('higgs_compression_ratio', 0.5)

    task_name = args['task_name']
    save_path = args['save_path']

    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = args['use_random_init']

    if use_random_init and skip_step_one:
        step2_lr = 5 * 1e-4

    print(f'[INFO] Using model: {model_name}')
    if use_compression and HIGGS_AVAILABLE:
        print(f'[INFO] HIGGS compression enabled with ratio: {compression_ratio}')

    # Update save name to include compression info
    compression_suffix = f"_higgs{compression_ratio}" if use_compression and HIGGS_AVAILABLE else ""
    
    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}-spectro{compression_suffix}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}-spectro{compression_suffix}'

    if use_random_init:
        save_name = 'randinit_' + save_name

    output_checkpoint_name_best = save_path + f'/best/{save_name}-spectro.pt'
    output_checkpoint_name_last = save_path + f'/last/{save_name}-spectro.pt'

    subject_choice = args['subjects']
    print(f'![Debug] Using {subject_choice}')
    eeg_type_choice = args['eeg_type']
    print(f'[INFO] EEG type {eeg_type_choice}')
    bands_choice = args['eeg_bands']
    print(f'[INFO] Using bands {bands_choice}')

    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    if torch.cuda.is_available():
        dev = args['cuda']
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f'[INFO] Using device {dev}')
    print()

    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-spectro.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset-spectro.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset-spectro.pickle'
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset-spectro.pickle'
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    print()

    """save config"""
    # Add compression settings to config
    args['use_higgs_compression'] = use_compression
    args['higgs_compression_ratio'] = compression_ratio
    
    with open(f'./config/decoding/{save_name}.json', 'w') as out_config:
        json.dump(args, out_config, indent=4)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name, 
        use_compression=use_compression, 
        compression_ratio=compression_ratio
    )

    # Move model to device
    model = model.to(device)

    # Create datasets
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject=subject_choice, setting=dataset_setting)
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject=subject_choice, setting=dataset_setting)
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice, setting=dataset_setting)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO] Train set size: ', len(train_set))
    print('[INFO] Dev set size: ', len(dev_set))
    print('[INFO] Test set size: ', len(test_set))

    # Save datasets with model-specific naming
    model_suffix = model_name.lower().replace('braintranslator', '')
    if use_compression and HIGGS_AVAILABLE:
        model_suffix += f"_higgs{compression_ratio}"
    
    with open(f"train_set_masked_raw_robert{model_suffix}.pkl", "wb") as f:
        pickle.dump(train_set, f)
    with open(f"dev_set_masked_raw_robert{model_suffix}.pkl", "wb") as f:
        pickle.dump(dev_set, f)
    with open(f"test_set_masked_raw_robert{model_suffix}.pkl", "wb") as f:
        pickle.dump(test_set, f)
    
    print(f"[INFO] Model setup complete. Using {'compressed' if use_compression and HIGGS_AVAILABLE else 'standard'} {model_name}")