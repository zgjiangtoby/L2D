import torch, os, sys, re, ordered_set, math, time
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import argparse
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sklearn.metrics import classification_report
import random
from collections import defaultdict, Counter
from transformers import (
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AdamW,
    AutoModelForCausalLM, AutoTokenizer
)
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

class DatasetReader(Dataset):
      def __init__(self, file_path, feature_columns, label_columns, id2label=None):

            self.id2label = id2label or self._default_id2label(label_columns)
            self.label2id = {v: k for k, v in self.id2label.items()}

            self.df = pd.read_csv(file_path, low_memory=False)
            
            if len(feature_columns) == 1: 
                  self.features = self.df.iloc[:, feature_columns].values
            elif len(feature_columns) == 2:  
                  self.features = list(zip(
                  self.df.iloc[:, feature_columns[0]].values,
                  self.df.iloc[:, feature_columns[1]].values
                  ))
            

            if isinstance(label_columns[0], int):  
                  self.labels = self.df.iloc[:, label_columns].values  

      def _default_id2label(self, label_columns):
            return {i: f"LABEL_{i}" for i in range(len(label_columns))}

      def __len__(self):
            return len(self.labels)

      def __getitem__(self, idx):
            
            if isinstance(self.features[0], tuple):  
                  text1, text2 = self.features[idx]
                  feature = (str(text1), str(text2))
            else:  
                  feature = str(self.features[idx][0])
                  
            label = self.labels[idx].item() if isinstance(self.labels[idx], np.ndarray) else self.labels[idx]
            assert label in self.id2label, f"Invalid label {label} at index {idx}. Valid labels: {self.id2label.keys()}"   

            
            return feature, label


class BaseRetriever:
      def __init__(self, train_dataset, id2label=None):

            if isinstance(train_dataset[0][0], tuple):
                  self.train_texts = [x[0] for x in train_dataset] 
                  self.is_pair_task = True
            else:
                  self.train_texts = [x[0] for x in train_dataset]
                  self.is_pair_task = False

            self.train_label_ids = [x[1] for x in train_dataset]  
            self.id2label = self._validate_id2label(id2label)
            self.label2id = {v:k for k,v in self.id2label.items()} 


            self.label_to_indices = defaultdict(list)
            for idx, label in enumerate(self.train_label_ids):
                  self.label_to_indices[label].append(idx) 
                        
      def _validate_id2label(self, id2label):
            if id2label is not None:
                  self.train_label_ids = [
                  tuple(label) if isinstance(label, np.ndarray) else label 
                  for label in self.train_label_ids]

                  unique_labels = set(self.train_label_ids)
                  missing = unique_labels - set(id2label.keys())
                  if missing:
                        print(f"id2label ID {missing}")
                  for m in missing:
                        id2label[m] = f"LABEL_{m}"
                  return id2label
            else:
                  unique_labels = sorted(set(self.train_label_ids))
                  return {i: f"LABEL_{i}" for i in unique_labels}

      def format_prompt(self, indices):
            examples = []
            for idx in indices:
                  if self.is_pair_task: 
                        text1, text2 = self.train_texts[idx]
                        examples.append({
                              "Premise": text1,
                              "Hypothesis": text2,
                              "Label": self.id2label[self.train_label_ids[idx]]
                        })
                  else:  
                        examples.append({
                              "Example": self.train_texts[idx],
                              "Label": self.id2label[self.train_label_ids[idx]]
                        })
            return examples
                  

      def _build_full_prompt(self, indices, test_text):
            ice_examples = self.format_prompt(indices)            
    
            
            if self.is_pair_task:  
                  premise, hypothesis = test_text
                  input_dict = {
                        "Premise": premise, 
                        "Hypothesis": hypothesis,
                        "Label": ""
                  }
            else:
                  input_dict = {"Input": test_text, "Label": ""}
            
            return ice_examples, input_dict
      

class SimilarityRetriever(BaseRetriever):
      def __init__(self, train_dataset, model_name, id2label=None):
            super().__init__(train_dataset, id2label)
            self.encoder = SentenceTransformer(model_name,trust_remote_code=True)
            self._precompute_embeddings()
            print("\nPrecomputing embeddings done........\n")
      
      def _precompute_embeddings(self):
           
            if self.is_pair_task:
                  sep_token = self.encoder.tokenizer.sep_token or "[SEP]"  

                  processed_texts = [
                        f"{t1}{sep_token}{t2}" for t1, t2 in self.train_texts
                  ]
            else:
                  processed_texts = self.train_texts
            
            self.embeddings = F.normalize(
                  torch.tensor(self.encoder.encode(processed_texts)),
                  p=2, dim=1
            )

      def retrieve(self, test_text, k=8):
      
            if self.is_pair_task and isinstance(test_text, tuple):
                  premise, hypothesis = test_text
                  processed_test = f"{premise}[SEP]{hypothesis}"  
            else:
                  processed_test = test_text  

          
            test_emb = F.normalize(
                  torch.tensor(self.encoder.encode([processed_test])),  
                  p=2, dim=1
            )[0]
      
            with torch.no_grad():
                  similarities = torch.mm(
                  test_emb.unsqueeze(0),
                  self.embeddings.T
                  )

            topk_values, topk_indices = torch.topk(
                  similarities, 
                  k=min(k, len(self.train_texts)),  
                  dim=1,
                  largest=True,
                  sorted=True  
            )
            self.similarities = topk_values
            
            return self._build_full_prompt(topk_indices.squeeze(), test_text)
      
      def _get_similarities(self):
            return self.similarities

class Finetune_XLM():
      def __init__(self, model_path, custom_id2label, num_labels, save_dir="./saved_slm", task_name=None): #!
            
            
            self.task_name = task_name
            self.num_labels = len(custom_id2label)
            print(self.task_name)
            self.save_path = os.path.join(save_dir, "best_{}_slm.pth".format(task_name))
            os.makedirs(save_dir, exist_ok=True)

            
            self.trained = False
            self.target_names = [custom_id2label[i] for i in sorted(custom_id2label.keys())]
            print(self.target_names)
           
            if os.path.exists(self.save_path):
                  print(f"Loading pre-trained model from {self.save_path}")
                  self._load_existing_model(model_path, num_labels)
                  self.trained = True 
            else:
                  self._init_new_model(model_path, num_labels)
          
            self.best_f1 = 0
            self.epochs_no_improve = 0
            self.early_stop = False
            

      def _init_new_model(self, model_path, num_labels):
         
            self.classifier = XLMRobertaForSequenceClassification.from_pretrained(
                  model_path, 
                  num_labels=num_labels,
                  problem_type="single_label_classification"
            )
            self.cls_tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
            self.optimizer = AdamW(self.classifier.parameters(), lr=2e-5)
            self.epochs = 10
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.classifier.to(self.device)

      def _load_existing_model(self, model_path, num_labels):
            
            self.classifier = XLMRobertaForSequenceClassification.from_pretrained(
                  model_path,
                  num_labels=num_labels,
                  state_dict=torch.load(self.save_path)
            )
            self.cls_tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.classifier.to(self.device)
            self.classifier.eval()
            print("Pretrained model loaded successfully")

      def _save_best_model(self):
            torch.save(self.classifier.state_dict(), self.save_path)
            self.epochs_no_improve = 0  
            print(f"Saved best model (F1: {self.best_f1:.4f}) to {self.save_path}")

      def _collate_fn(self, batch):
            
            text_pairs, labels = zip(*batch)
            text1_batch = [pair[0] for pair in text_pairs]
            text2_batch = [pair[1] for pair in text_pairs]
            
      
            return (text1_batch, text2_batch), torch.LongTensor(labels)
      def _tokenize(self, texts):
            
            if isinstance(texts, tuple) and len(texts) == 2:
                  
                  return self.cls_tokenizer(
                        texts[0], 
                        texts[1],
                        padding='max_length',
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                  ).to(self.device)
            else:
                  
                  return self.cls_tokenizer(
                        texts,
                        padding='max_length',
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                  ).to(self.device)

      def tokenize_text(self, texts):
            
            if isinstance(texts, tuple) and len(texts) == 2:
                  return self._tokenize(texts)
            elif isinstance(texts, str):
                  return self._tokenize(texts)
            else:
                  return [self._tokenize(text) for text in texts]


      def _evaluate(self, dataloader):
            self.classifier.eval()
            total_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                  for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                        texts, labels = batch
                        inputs = self._tokenize(texts).to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = self.classifier(**inputs, labels=labels)
                        loss = outputs.loss
                        total_loss += loss.item()
                        
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        preds = torch.argmax(probs, dim=1)
                        
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                  
                  report = classification_report(
                        all_labels, all_preds, 
                        target_names=self.target_names,
                        output_dict=True
                  )
                  return {
                        'loss': total_loss / len(dataloader),
                        'accuracy': report['accuracy'],
                        'f1': report['weighted avg']['f1-score']  
                  }

      def _forward(self, train_set, val_set):
            if self.trained:
                  print("done")
                  return 0

           
            total_training_start_time = time.time()

            
            if self.task_name in ['mnli', 'qnli']:
                  train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=self._collate_fn)
                  val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=self._collate_fn)
            else:
                  train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
                  val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

            for epoch in range(self.epochs):
                  if self.early_stop:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break


                  
                  self.classifier.train()
                  epoch_loss = 0
                  progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
                  
                  for batch in progress_bar:
                        texts, labels = batch
                        inputs = self._tokenize(texts).to(self.device)
                        labels = labels.to(self.device)
                        
                        self.optimizer.zero_grad()
                        outputs = self.classifier(**inputs, labels=labels)
                        loss = outputs.loss
                        loss.backward()
                        self.optimizer.step()
                        
                        epoch_loss += loss.item()
                        progress_bar.set_postfix({'train_loss': loss.item()})

                        
                  eval_metrics = self._evaluate(val_loader)
                  print(f"\nEpoch {epoch+1} | "
                        f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
                        f"Val Loss: {eval_metrics['loss']:.4f} | "
                        f"Val Acc: {eval_metrics['accuracy']:.4f} | "
                        f"Val F1: {eval_metrics['f1']:.4f}")

                  
                  if eval_metrics['f1'] > self.best_f1:
                        self.best_f1 = eval_metrics['f1']
                        self._save_best_model()
                  else:
                        self.epochs_no_improve += 1
                        if self.epochs_no_improve >= 2:
                              self.early_stop = True

            
            if os.path.exists(self.save_path):
                  self.classifier.load_state_dict(torch.load(self.save_path))
                  print(f"Final model loaded with F1: {self.best_f1:.4f}")
            
            
            total_training_time = time.time() - total_training_start_time
            print(f"total_training_time: {total_training_time:.4f}s")
            return total_training_time

class L2DRetriever(SimilarityRetriever):
      def __init__(self, train_dataset, retriever_model, classifier_model, id2label=None, device=None, args=None):
            
            init_start_time = time.time()
            
            super().__init__(train_dataset, retriever_model, id2label)
            self.classifier = classifier_model
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.args = args
            
            
            self.total_time = 0
            self.fine_tune_time = 0
            self.label_distribution_time = 0
            self.retrieve_time = 0
            self.llm_inference_time = 0
            
            
            label_dist_start = time.time()
            self._precompute_label_distributions()
            self.label_distribution_time = time.time() - label_dist_start
            
            
            self.init_time = time.time() - init_start_time
            print(f"\nL2DRetriever_init_time: {self.init_time:.4f}s")
            print(f"label_distribution_time: {self.label_distribution_time:.4f}s")
            print("\nLabel distributions precomputed........\n")

      def _precompute_label_distributions(self):
            
            self.label_probs = []
            self.classifier.classifier.eval()
            with torch.no_grad():
                  for text in tqdm(self.train_texts, desc="Precomputing label probs"):
                        inputs = self.classifier.tokenize_text(text).to(self.device)
                        outputs = self.classifier.classifier(**inputs)
                        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
                        self.label_probs.append(probs)
            self.label_probs = np.array(self.label_probs)

      def _kl_divergence(self, p, q):
            
            epsilon = 1e-10
            p = np.clip(p, epsilon, 1)
            q = np.clip(q, epsilon, 1)
            return np.sum(p * np.log(p / q))

      
      def retrieve(self, test_text, n=8):
            
            retrieve_start_time = time.time()
            
            
            k = self.args.candidate
            
            ice_examples, input_dict = super().retrieve(test_text, k=k)
            semantic_scores = super()._get_similarities().cpu().numpy().tolist()[0]

            
            with torch.no_grad():
                  if isinstance(test_text, tuple) and len(test_text) == 2:
                        premise, hypothesis = test_text
                        test_inputs = self.classifier.tokenize_text((premise, hypothesis))
                  else:
                        test_inputs = self.classifier.tokenize_text(test_text)
                  
                  test_outputs = self.classifier.classifier(**test_inputs)
                  test_prob = F.softmax(test_outputs.logits, dim=-1).cpu().numpy()[0]
            
            
            distribution_scores = []
            for idx_dict, sim_score in zip(ice_examples, semantic_scores):
                  
                  if self.is_pair_task:  
                        example_text = (idx_dict['Premise'], idx_dict['Hypothesis'])
                  else:  
                        example_text = idx_dict['Example']
                  
                  try:
                        idx = self.train_texts.index(example_text)
                  except ValueError:
                        print(f"Text not found in train_texts: {example_text}")
                        continue
                  
                  train_prob = self.label_probs[idx]
                  
                  
                  m = 0.5 * (test_prob + train_prob)
                  js_div = 0.5 * self._kl_divergence(test_prob, m) + 0.5 * self._kl_divergence(train_prob, m)
                  dis_scores = 1 - js_div
                  
                  alpha = self.args.alpha  
                  hybrid_scores = alpha * sim_score + (1-alpha) * dis_scores
                  distribution_scores.append((idx, hybrid_scores))  
            
            
            sorted_indices = sorted(distribution_scores, key=lambda x: x[1], reverse=True)
            final_indices = [x[0] for x in sorted_indices[:n]]
            
            
            self.retrieve_time += time.time() - retrieve_start_time
            
            return self._build_full_prompt(final_indices, test_text)
            
class LLMEvaluator:
      def __init__(self, model_path, custom_id2label, max_new_tokens=50 ):
            self.devices = [torch.device(f"cuda:{i}") for i in range(2)]  
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                                           trust_remote_code=True, 
                                                           padding_side="left")
                                       
            
            self.custom_id2label = custom_id2label
            self.model = AutoModelForCausalLM.from_pretrained(
                  model_path, 
                  torch_dtype=torch.bfloat16,
                  trust_remote_code=True,
                  device_map="balanced", 
                  max_memory={i: "24GiB" for i in range(2)},  
                  local_files_only=True
            ).eval()

           
             
            if self.tokenizer.pad_token is None:
                  self.tokenizer.pad_token = self.tokenizer.eos_token  
                  self.model.config.pad_token_id = self.tokenizer.pad_token_id = self.tokenizer.eos_token_id = 128001 
            if torch.cuda.device_count() > 1:
                  self.model = torch.nn.DataParallel(self.model)
            
           

            self.max_new_tokens = max_new_tokens
      


      def build_icl_prompt(self, examples, task_name):
            icl_prompt_template = """
                  You're a classifier based on the following rules:
                  1. Output only the final Label, format: {custom_id2label}
                  2. Prohibit any process of interpretation or reasoning
                  
                  Determine the categories of input text based on the following examples:：
                  {examples}
                  """
            
            if task_name in ["sst2", "sst5", "cr"]:
                  example_str = "\n\n".join([
                        f"In-context Example{i+1}:\n Review:{ex['Example']}\n Sentiment:{ex['Label']}" 
                        for i, ex in enumerate(examples)
                  ])
            elif task_name == "subj":
                  example_str = "\n\n".join([
                        f"In-context Example{i+1}:\n Input:{ex['Example']}\n Type:{ex['Label']}" 
                        for i, ex in enumerate(examples)
                  ])
            elif task_name == "ag_news":
                  example_str = "\n\n".join([
                        
                        f"{ex['Example']}. What is this text about? World, Sports, or Technology?. {ex['Label']}"
                        for i, ex in enumerate(examples)
                  ])
            elif task_name == "mnli":
                  example_str = "\n\n".join([
                        f"In-context Example{i+1}:\n {ex['Premise']}\n Can we know {ex['Hypothesis']}? {ex['Label']}" 
                        for i, ex in enumerate(examples)
                  ])
            else:
                  example_str = "\n\n".join([
                        f"In-context Example{i+1}:\n {ex['Premise']}\n Can we know {ex['Hypothesis']}? {ex['Label']}" 
                        for i, ex in enumerate(examples)
                  ])
            
            return icl_prompt_template.format(custom_id2label=[v.lower() for v in custom_id2label.values()][0] + " or " + [v.lower() for v in custom_id2label.values()][1],
                                               examples=example_str)
                  

      def generate_response(self, examples, query, task_name):
            

            system_prompt = self.build_icl_prompt(examples, task_name)
            messages = [{"role": "user", "content": system_prompt}]
            
            
            if task_name not in ["sst2", "sst5", "cr", "subj", "ag_news"]:
                  user_content = f"{query['Premise']}\n Can we know {query['Hypothesis']}? \n Label:"
            else:
                  user_content = query["Input"] if isinstance(query, dict) else str(query)
            
            messages.append({"role": "assistant", "content": user_content})  

            
            generation_config = {
                  "max_new_tokens": self.max_new_tokens,
                  "temperature": 0.2,         
                  "do_sample": True,
                  "pad_token_id": self.tokenizer.eos_token_id
            }                       
            
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(
                  input_text,
                  return_tensors="pt", 
                  max_length=4096, 
                  truncation=True
            ).to(self.devices[0])
            
            with torch.amp.autocast('cuda'), torch.no_grad():
                  generated_ids = self.model.module.generate(**inputs,
                        **generation_config
                        )

                  generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
                  response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


                  labels = [v.lower() for v in self.custom_id2label.values()]
                  label_pattern = r'[\s\S]*?({})[\s\S]*?'.format("|".join(map(re.escape, labels)))
                  label_match = re.search(label_pattern, response, flags=re.IGNORECASE)  
                  
                  return label_match.group(0) if label_match else "Label: Unknown" 

class LLMEvaluation:
      def __init__(self, id2label, task_type='classification'):
            self.id2label = id2label
            self.task_type = task_type

            
      def _extract_prediction(self, response):
            
            for label_id, label_text in self.id2label.items():
                  if label_text.lower() in response.lower():
                        return label_id
            return None 
      
      def compute_metrics(self, pred_labels, gold_labels):
            
            metrics = {}
            
            if self.task_type == 'classification':
                  metrics['accuracy'] = accuracy_score(gold_labels, pred_labels)
                  metrics['f1_macro'] = f1_score(gold_labels, pred_labels, average='macro')
                  metrics['class_report'] = classification_report(
                  gold_labels, pred_labels, 
                  target_names=list(self.id2label.values())
                  )
            
            return metrics
      
      def evaluate_batch(self, responses, gold_labels):            
            
            pred_labels = [self._extract_prediction(r) for r in responses]
            valid_idx = [i for i, p in enumerate(pred_labels) if p is not None]

           
            filtered_pred = [pred_labels[i] for i in valid_idx]
            filtered_gold = [gold_labels[i] for i in valid_idx]
            if not filtered_pred:
                  return {"error": "No valid predictions"}
                  
            return self.compute_metrics(filtered_pred, filtered_gold)

def create_split_datasets(dataset, split_ratio=0.8, seed=521):
      
      train_size = int(len(dataset) * split_ratio)
      test_size = len(dataset) - train_size
      
      
      generator = torch.Generator().manual_seed(seed)
      return random_split(
            dataset, 
            [train_size, test_size],
            generator=generator
      )

def inferencer(test_text, retriever, task_name, method, output_dir):
      test_texts = test_text
      all_responses = []
      
      
      llm_inference_start = time.time()
      
      for text in tqdm(test_texts, desc="Generating Responses"):
            
            single_inference_start = time.time()
                              
            ice, input_text = retriever.retrieve(text)

            
            response = llm.generate_response(ice, input_text, task_name)
            
            
            if hasattr(retriever, 'llm_inference_time'):
                  retriever.llm_inference_time += time.time() - single_inference_start
            
            all_responses.append(response)
      
      
      total_llm_time = time.time() - llm_inference_start
      
      
      if isinstance(retriever, Dual_topkRetriever):
            retriever.total_time = retriever.init_time + retriever.fine_tune_time + retriever.label_distribution_time + retriever.retrieve_time + total_llm_time

            timing_file = os.path.join(output_dir, f"timing_{method}.json")
            with open(timing_file, 'w', encoding='utf-8') as f:
                  import json
                  json.dump({
                        "method": method,
                        "fine_tune_time": retriever.fine_tune_time,
                        "label_distribution_time": retriever.label_distribution_time,
                        "retrieve_time": retriever.retrieve_time,
                        "llm_inference_time": total_llm_time,
                        "total_time": retriever.total_time,
                        "num_samples": len(test_texts)
                  }, f, indent=2)
      
      return all_responses


if __name__ == "__main__":
      

      parser = argparse.ArgumentParser(description='Retriever Experiments')
     
      parser.add_argument('--train_path', type=str, required=True)
      parser.add_argument('--test_path', type=str, required=True)
      parser.add_argument('--output_dir', type=str, default='./results')
      parser.add_argument('--method', type=str, default='random')
      parser.add_argument('--task_name', type=str, default='sst2')
      
      
      parser.add_argument('--retriever_model', type=str, default='gte-multilingual-base')
      parser.add_argument('--label_model', type=str, default='xlm-roberta-base')
      parser.add_argument('--llm_model', type=str, default='Qwen2.5-7B-Instruct')
      
     
      parser.add_argument('--fine_tune', action='store_const', const=True, default=False, help='True')
      parser.add_argument('--candidate', type=int, default=30)


      parser.add_argument('--seed', type=int, default=521)
      parser.add_argument('--alpha', type=float, default=0.5)
      parser.add_argument('--select_time', type=int, default=5)


      args = parser.parse_args()

      if args.task_name == "sst2":
            custom_id2label = {
                  0: "positive",#"positive",
                  1: "negative" #"negative"
            }
            train_dataset = DatasetReader(args.train_path, [0], [1], id2label=custom_id2label)
            test_dataset = DatasetReader(args.test_path, [0], [1], id2label=custom_id2label)
      elif args.task_name == "sst5":
            custom_id2label = {
                  0: "terrible",
                  1: "bad",
                  2: "okay",
                  3: "good",
                  4: "great",
            }
            train_dataset = DatasetReader(args.train_path, [0], [1], id2label=custom_id2label)
            test_dataset = DatasetReader(args.test_path, [0], [1], id2label=custom_id2label)
      elif args.task_name == "subj":
            custom_id2label = {
                  0: "objective", #"objective",
                  1: "subjective" #"subjective"
            }
            train_dataset = DatasetReader(args.train_path, [0], [1], id2label=custom_id2label)
            test_dataset = DatasetReader(args.test_path, [0], [1], id2label=custom_id2label)
      elif args.task_name == "cr":
            custom_id2label = {
                  0: "negative", #"negative",
                  1: "positive" #"positive"
            }
            train_dataset = DatasetReader(args.train_path, [0], [1], id2label=custom_id2label)
            test_dataset = DatasetReader(args.test_path, [0], [1], id2label=custom_id2label)
      elif args.task_name == "ag_news":
            custom_id2label = {
                  0: "World",
                  1: "Sports",
                  2: "Business",
                  3: "Technology"
            }
            train_dataset = DatasetReader(args.train_path, [0], [1], id2label=custom_id2label)
            test_dataset = DatasetReader(args.test_path, [0], [1], id2label=custom_id2label)
      elif args.task_name == "mnli":
            custom_id2label = {
                  0: "Entailment",
                  1: "Neutral",
                  2: "Contradiction",
            }
            train_dataset = DatasetReader(args.train_path, [0, 1], [2], id2label=custom_id2label)
            test_dataset = DatasetReader(args.test_path, [0,1], [2], id2label=custom_id2label)
      else:
            custom_id2label = {
                  0: "Entailment",
                  1: "Contradiction",
            }
            train_dataset = DatasetReader(args.train_path, [0, 1], [2], id2label=custom_id2label)
            test_dataset = DatasetReader(args.test_path, [0, 1], [2], id2label=custom_id2label)

      num_labels = len(custom_id2label)


      if args.method == "L2D":
            if args.fine_tune:
                  f_train, f_val = create_split_datasets(train_dataset, split_ratio=0.9, seed=args.seed)
                  classifier = Finetune_XLM(args.label_model, custom_id2label=custom_id2label, num_labels=num_labels,
                                            task_name=args.task_name)
                  
                  fine_tune_time = classifier._forward(f_train, f_val)
                  cls_model = classifier.classifier

            retriever = Dual_topkRetriever(train_dataset, args.retriever_model, classifier, id2label=custom_id2label, args=args)
            
            if 'fine_tune_time' in locals():
                  retriever.fine_tune_time = fine_tune_time

   
      test_texts = [test_dataset[i][0] for i in range(len(test_dataset))]
      gold_labels = [test_dataset[i][1] for i in range(len(test_dataset))]

      llm = LLMEvaluator(args.llm_model, custom_id2label) 
      evaluator = LLMEvaluation(custom_id2label, task_type='classification')

      all_responses = inferencer(test_text=test_texts, retriever=retriever, task_name=args.task_name, method = args.method, output_dir = args.output_dir)

      metrics = evaluator.evaluate_batch(all_responses, gold_labels) 
      

      if not os.path.exists(args.output_dir):  
            os.mkdir(args.output_dir)
            
      with open(args.output_dir + 'result_new_{}_{}.txt'.format(args.candidate, args.method), 'w', encoding='utf-8') as f:
            
            original_stdout = sys.stdout
            sys.stdout = f
            
            
            print("\n=== Evaluation Report ===")
            print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"F1 Macro: {metrics.get('f1_macro', 0):.4f}")
            if 'class_report' in metrics:
                  print("\nClassification Report:")
                  print(metrics['class_report'])
            
            
            sys.stdout = original_stdout
