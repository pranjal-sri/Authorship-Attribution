import torch
from torch.utils.data import Dataset
import json
from collections import defaultdict
import random
from rst_tree.serializer import TreeSerializer
from rst_tree.tree_attention_extractor import TreeAttentionSpanExtractor

class Author_file_pair_generator:
    def __init__(self, author_files):
        self.author_files = author_files
        self.pairs = [(i, j) for i in range(len(author_files))
                     for j in range(i+1, len(author_files))]
        random.shuffle(self.pairs)
        self.pointer = 0

    def __len__(self):
        return len(self.pairs)

    def get_next_pair(self):
        if self.pointer >= len(self.pairs):
            random.shuffle(self.pairs)
            self.pointer = 0

        file_a, file_b = self.pairs[self.pointer]
        self.pointer += 1
        return self.author_files[file_a], self.author_files[file_b]

class ReutersRSTDataset(Dataset):
    def __init__(self, tokenizer, author_files, base_dir_path,
                 MAX_EPISODE_LENGTH=64, MAX_EPISODES=16, batch_size=32):
        self.authors = list(author_files.keys())

        assert batch_size <= len(self.authors), "Batch size cannot be greater than number of authors"
        self.batch_size = batch_size
        self.base_dir_path = base_dir_path
        self.tokenizer = tokenizer
        self.MAX_EPISODE_LENGTH = MAX_EPISODE_LENGTH
        self.MAX_EPISODES = MAX_EPISODES

        self.author_file_pairs_map = {}
        for author in author_files:
            self.author_file_pairs_map[author] = Author_file_pair_generator(author_files[author])

        self.n_samples_per_author = len(self.author_file_pairs_map[self.authors[0]])

        self.authors_data = self.generate_data()

    def generate_data(self):
        authors_data = []
        for _ in range(0, self.n_samples_per_author * len(self.authors), self.batch_size):
            authors_data.append(random.choices(self.authors, k=self.batch_size))
        authors_data.append(random.choices(self.authors, k=self.batch_size))
        return authors_data

    def __len__(self):
        return self.n_samples_per_author * len(self.authors)

    def __getitem__(self, idx):
        author = self.authors_data[idx]
        file_a, file_b = self.author_file_pairs_map[author].get_next_pair()
        return ([file_a, file_b], self.get_file_tensors(file_a), self.get_file_tensors(file_b))

    def get_file_tensors(self, file):
        with open(f"{self.base_dir_path}/{file}.txt") as f:
            file_dict = json.loads(f.read())

        filtered_chunk_keys = list(file_dict.keys())[: min(self.MAX_EPISODES, len(file_dict))]

        episode_input_ids = torch.zeros((self.MAX_EPISODES, self.MAX_EPISODE_LENGTH), dtype=torch.long)
        episode_attention_masks_encoder = torch.zeros((self.MAX_EPISODES, self.MAX_EPISODE_LENGTH))
        episode_attention_masks_granular = torch.zeros((self.MAX_EPISODES, 6, self.MAX_EPISODE_LENGTH, self.MAX_EPISODE_LENGTH))

        for i, k in enumerate(filtered_chunk_keys):
            tree, height_map, tree_height = TreeSerializer.dict_to_tree(file_dict[k]['tree'])
            attn_spans = TreeAttentionSpanExtractor.extract_attention_spans(tree, height_map, tree_height)
            text = file_dict[k]['text']

            enc = self.tokenizer(text, return_tensors='pt', max_length=self.MAX_EPISODE_LENGTH,
                               truncation=True, padding='max_length')
            attention_mask = generate_attention_mask(enc, attn_spans, len(text))
            episode_input_ids[i] = enc.input_ids
            episode_attention_masks_encoder[i] = enc.attention_mask
            episode_attention_masks_granular[i] = attention_mask

        episode_attention_mask = torch.zeros((self.MAX_EPISODES,))
        episode_attention_mask[: min(self.MAX_EPISODES, len(file_dict))] = 1.0

        return {
            'input_ids': episode_input_ids,
            'attention_mask_episodes': episode_attention_mask,
            'attention_masks_encoder': episode_attention_masks_encoder,
            'attention_masks_granular': episode_attention_masks_granular,
        }

def generate_attention_mask(encoding, attention_spans, text_len):
    total_tokens = len(encoding.tokens())

    if encoding.token_to_chars(len(encoding.input_ids[0])-2):
        end_point = encoding.token_to_chars(len(encoding.input_ids[0])-2).end
    else:
        end_point = text_len

    total_layers = len(attention_spans)
    attention_mask = torch.zeros((total_layers, total_tokens, total_tokens))

    for layer, char_spans in attention_spans.items():
        for i, (start_char, end_char) in enumerate(char_spans):
            start_token = encoding.char_to_token(start_char)
            end_token = encoding.char_to_token(end_char - 1)

            if i == 0:
                start_token = 0
            end_flag = False
            if start_char < end_point and end_char >= end_point:
                end_token = total_tokens-1
                end_flag = True

            if start_token is not None and end_token is not None:
                attention_mask[layer, start_token:end_token + 1, start_token:end_token + 1] = 1

    return attention_mask