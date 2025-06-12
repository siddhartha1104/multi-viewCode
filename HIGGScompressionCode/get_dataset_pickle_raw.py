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

# Updated imports for Gemma/Llama + original models
from transformers import (
    GemmaTokenizer, GemmaForCausalLM,
    LlamaTokenizer, LlamaForCausalLM,
    BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig,
    BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, 
    RobertaTokenizer, RobertaForSequenceClassification
)

# HIGGS compression (with fallback)
try:
    from higgs import HIGGSCompressor
    HIGGS_AVAILABLE = True
except ImportError:
    print("[WARNING] HIGGS not available, using standard models")
    HIGGS_AVAILABLE = False

# Dynamic data module import based on configuration
def import_data_module(data_type):
    """Dynamically import the appropriate data module"""
    if data_type == 'raw':
        from data_raw import ZuCo_dataset
    elif data_type == 'masked_raw_all':
        from data_masked_raw_all import ZuCo_dataset
    elif data_type == 'masked_raw_robert':
        from data_masked_raw_robert import ZuCo_dataset
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    return ZuCo_dataset

from model_decoding_spectro import BrainTranslator, BrainTranslatorNaive
from config import get_config


class EnhancedBrainTranslator(nn.Module):
    """Enhanced Brain Translator with support for multiple LLMs and HIGGS compression"""
    
    def __init__(self, model_name, compression_ratio=0.5, use_compression=True, 
                 brain_feature_dim=1024, hidden_dim=512):
        super().__init__()
        self.model_name = model_name
        self.use_compression = use_compression and HIGGS_AVAILABLE
        self.brain_feature_dim = brain_feature_dim
        self.hidden_dim = hidden_dim
        
        # Load and setup language model
        self.tokenizer, self.language_model = self._setup_language_model(compression_ratio)
        
        # Brain signal processing components
        self.brain_encoder = self._build_brain_encoder()
        self.projection_layer = self._build_projection_layer()
        
        # Additional components for enhanced architecture
        if 'enhanced' in model_name.lower():
            self.attention_bridge = self._build_attention_bridge()
            self.feature_fusion = self._build_feature_fusion()
    
    def _setup_language_model(self, compression_ratio):
        """Setup language model with optional compression"""
        if 'gemma' in self.model_name.lower():
            tokenizer = GemmaTokenizer.from_pretrained('google/gemma-2b-it')
            base_model = GemmaForCausalLM.from_pretrained('google/gemma-2b-it')
            
        elif 'llama' in self.model_name.lower():
            tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
            base_model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
            
        elif 'bart' in self.model_name.lower() or self.model_name in ['BrainTranslator', 'BrainTranslatorNaive']:
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            base_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
            
        elif 'bert' in self.model_name.lower():
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            config = BertConfig.from_pretrained("bert-base-cased")
            config.is_decoder = True
            base_model = BertLMHeadModel.from_pretrained("bert-base-cased", config=config)
            
        else:
            # Default to BART
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            base_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        
        # Apply HIGGS compression if available and requested
        if self.use_compression:
            print(f"[INFO] Applying HIGGS compression with ratio {compression_ratio}")
            compressor = HIGGSCompressor(compression_ratio=compression_ratio)
            compressed_model = compressor.compress(base_model)
            return tokenizer, compressed_model
        
        return tokenizer, base_model
    
    def _build_brain_encoder(self):
        """Build brain signal encoder with residual connections"""
        return nn.Sequential(
            nn.Linear(self.brain_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU()
        )
    
    def _build_projection_layer(self):
        """Project brain embeddings to language model dimension"""
        lm_hidden_size = self.language_model.config.hidden_size
        return nn.Sequential(
            nn.Linear(self.hidden_dim // 2, lm_hidden_size),
            nn.LayerNorm(lm_hidden_size),
            nn.Tanh()
        )
    
    def _build_attention_bridge(self):
        """Build attention mechanism between brain and language representations"""
        lm_hidden_size = self.language_model.config.hidden_size
        return nn.MultiheadAttention(
            embed_dim=lm_hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
    
    def _build_feature_fusion(self):
        """Build feature fusion layer"""
        lm_hidden_size = self.language_model.config.hidden_size
        return nn.Sequential(
            nn.Linear(lm_hidden_size * 2, lm_hidden_size),
            nn.LayerNorm(lm_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, brain_signals, target_ids=None, attention_mask=None):
        # Encode brain signals
        brain_embeddings = self.brain_encoder(brain_signals)
        projected_embeddings = self.projection_layer(brain_embeddings)
        
        # Enhanced architecture with attention bridge
        if hasattr(self, 'attention_bridge'):
            # Apply attention between brain and language representations
            attended_features, _ = self.attention_bridge(
                projected_embeddings, projected_embeddings, projected_embeddings
            )
            
            # Fuse original and attended features
            fused_embeddings = self.feature_fusion(
                torch.cat([projected_embeddings, attended_features], dim=-1)
            )
            final_embeddings = fused_embeddings
        else:
            final_embeddings = projected_embeddings
        
        # Generate text using language model
        outputs = self.language_model(
            inputs_embeds=final_embeddings,
            attention_mask=attention_mask,
            labels=target_ids
        )
        
        return outputs


def setup_model_and_tokenizer(model_name, use_compression=True, compression_ratio=0.5, 
                             brain_feature_dim=1024):
    """Setup model and tokenizer with enhanced flexibility"""
    
    # Enhanced models with multiple LLM support
    if any(llm in model_name.lower() for llm in ['gemma', 'llama', 'enhanced']):
        model = EnhancedBrainTranslator(
            model_name=model_name,
            use_compression=use_compression,
            compression_ratio=compression_ratio,
            brain_feature_dim=brain_feature_dim
        )
        tokenizer = model.tokenizer
        
    # Original models
    elif model_name in ['BrainTranslator', 'BrainTranslatorNaive']:
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

    ''' Enhanced config parameters '''
    dataset_setting = 'unique_sent'
    
    # Data processing type - determines which data module to use
    data_type = args.get('data_type', 'raw')  # 'raw', 'masked_raw_all', 'masked_raw_robert'
    
    # Model configuration
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    batch_size = args['batch_size']
    model_name = args['model_name']
    
    # Enhanced model options:
    # model_name = 'BrainTranslatorGemma'
    # model_name = 'BrainTranslatorLlama'  
    # model_name = 'EnhancedBrainTranslatorGemma'
    # model_name = 'EnhancedBrainTranslatorLlama'
    
    # HIGGS compression settings
    use_compression = args.get('use_higgs_compression', True)
    compression_ratio = args.get('higgs_compression_ratio', 0.5)
    brain_feature_dim = args.get('brain_feature_dim', 1024)
    
    task_name = args['task_name']
    save_path = args['save_path']
    
    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = args['use_random_init']

    if use_random_init and skip_step_one:
        step2_lr = 5 * 1e-4

    print(f'[INFO] Using model: {model_name}')
    print(f'[INFO] Data processing type: {data_type}')
    if use_compression and HIGGS_AVAILABLE:
        print(f'[INFO] HIGGS compression enabled with ratio: {compression_ratio}')

    # Enhanced save naming with data type and compression info
    data_suffix = f"_{data_type}" if data_type != 'raw' else ""
    compression_suffix = f"_higgs{compression_ratio}" if use_compression and HIGGS_AVAILABLE else ""
    
    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}-spectro{data_suffix}{compression_suffix}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}-spectro{data_suffix}{compression_suffix}'

    if use_random_init:
        save_name = 'randinit_' + save_name

    output_checkpoint_name_best = save_path + f'/best/{save_name}-spectro.pt'
    output_checkpoint_name_last = save_path + f'/last/{save_name}-spectro.pt'

    # EEG configuration
    subject_choice = args['subjects']
    print(f'![Debug] Using subjects: {subject_choice}')
    eeg_type_choice = args['eeg_type']
    print(f'[INFO] EEG type: {eeg_type_choice}')
    bands_choice = args['eeg_bands']
    print(f'[INFO] Using bands: {bands_choice}')

    ''' Set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' Set up device '''
    if torch.cuda.is_available():
        dev = args['cuda']
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f'[INFO] Using device: {dev}')

    ''' Dynamic data loading '''
    # Import appropriate data module
    ZuCo_dataset = import_data_module(data_type)
    
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

    """Enhanced config saving"""
    enhanced_args = args.copy()
    enhanced_args.update({
        'data_type': data_type,
        'use_higgs_compression': use_compression,
        'higgs_compression_ratio': compression_ratio,
        'brain_feature_dim': brain_feature_dim,
        'higgs_available': HIGGS_AVAILABLE
    })
    
    with open(f'./config/decoding/{save_name}.json', 'w') as out_config:
        json.dump(enhanced_args, out_config, indent=4)

    # Setup enhanced model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name=model_name,
        use_compression=use_compression,
        compression_ratio=compression_ratio,
        brain_feature_dim=brain_feature_dim
    )

    # Move model to device
    model = model.to(device)

    # Create datasets
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, 
                            subject=subject_choice, setting=dataset_setting)
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, 
                          subject=subject_choice, setting=dataset_setting)
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, 
                           subject=subject_choice, setting=dataset_setting)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO] Dataset sizes:')
    print(f'  Train: {len(train_set)}')
    print(f'  Dev: {len(dev_set)}')
    print(f'  Test: {len(test_set)}')

    # Enhanced dataset saving with comprehensive naming
    file_suffix = f"{data_type}"
    if 'enhanced' in model_name.lower():
        file_suffix += "_enhanced"
    if any(llm in model_name.lower() for llm in ['gemma', 'llama']):
        llm_type = 'gemma' if 'gemma' in model_name.lower() else 'llama'
        file_suffix += f"_{llm_type}"
    if use_compression and HIGGS_AVAILABLE:
        file_suffix += f"_higgs{compression_ratio}"
    
    dataset_files = {
        'train': f"train_set_{file_suffix}.pkl",
        'dev': f"dev_set_{file_suffix}.pkl", 
        'test': f"test_set_{file_suffix}.pkl"
    }
    
    for split, dataset in [('train', train_set), ('dev', dev_set), ('test', test_set)]:
        with open(dataset_files[split], "wb") as f:
            pickle.dump(dataset, f)
        print(f"[INFO] Saved {split} dataset to {dataset_files[split]}")
    
    print(f"\n[INFO] Setup complete!")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Data type: {data_type}")
    print(f"[INFO] Compression: {'Enabled' if use_compression and HIGGS_AVAILABLE else 'Disabled'}")
    print(f"[INFO] Total parameters: {sum(p.numel() for p in model.parameters()):,}")