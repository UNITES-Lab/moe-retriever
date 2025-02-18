import datetime
import os
import torch
import numpy as np
import argparse
import random
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from copy import deepcopy
from tqdm import trange
from models import MoE_Retriever, FusionLayer
from data import load_and_preprocess_data, load_and_preprocess_data_mimic, create_loaders, process_2d_to_3d
import logging
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")

# Set random seed for reproducibility
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(log_path, log_name, file_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    file_handler = logging.FileHandler(os.path.join(log_path, file_name))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# Utility function to convert string to bool
def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

# Parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='MoE-Retriever')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data', type=str, default='adni')
    parser.add_argument('--modality', type=str, default='IG') # I G C B for ADNI, L N C for MIMIC
    parser.add_argument('--num_candidates', type=int, default=8) # Number of candidates in each modality bank
    parser.add_argument('--num_candidates_shared', type=int, default=4) # Number of candidates in each modality bank
    parser.add_argument('--num_supporting_samples', type=int, default=4) # Number of supporting samples
    parser.add_argument('--initial_filling', type=str, default='mean') # None mean
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--warm_up_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers_enc', type=int, default=1) # Number of MLP layers for encoders
    parser.add_argument('--num_layers_fus', type=int, default=1) # Number of MLP layers for fusion model
    parser.add_argument('--num_layers_pred', type=int, default=1) # Number of MLP layers for fusion model
    parser.add_argument('--num_heads', type=int, default=4) # Number of heads
    parser.add_argument('--num_workers', type=int, default=4) # Number of workers for DataLoader
    parser.add_argument('--pin_memory', type=str2bool, default=True) # Pin memory in DataLoader
    parser.add_argument('--use_common_ids', type=str2bool, default=False) # Use common ids across modalities
    parser.add_argument('--patch', type=str2bool, default=True) # Use common ids across modalities
    parser.add_argument('--num_patches', type=int, default=16) # Use common ids across modalities
    parser.add_argument('--num_experts', type=int, default=32) # Number of Experts
    parser.add_argument('--num_routers', type=int, default=1) # Number of Routers
    parser.add_argument('--fusion_sparse', type=str2bool, default=False) # Whether to include SMoE in Fusion Layer
    parser.add_argument('--top_k', type=int, default=4) # Number of Routers
    parser.add_argument('--dropout', type=float, default=0.5) # Number of Routers
    parser.add_argument('--gate_loss_weight', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--save', type=str2bool, default=False) # Use common ids across modalities

    return parser.parse_known_args()

def contains_all_chars(sub, string):
    return all(char in string for char in sub)

def get_supporting_group(missing_modality=None, observed_modalities=None, mc_idx_dict=None, num_samples=2):
    # Find all samples in the dataset with the required observed modality
    sufficient_modalities = missing_modality + observed_modalities
    all_set = set(mc_idx_dict.keys())
    available_mcs = [mc for mc in all_set if contains_all_chars(sufficient_modalities, mc)]
    idx_pool = [item for key in available_mcs for item in mc_idx_dict[key]]

    return random.sample(idx_pool, num_samples)

def train_and_evaluate(args, seed):
    seed_everything(seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    num_modalities = len(args.modality)

    if args.data == 'adni':
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, mc_idx_dict, num_mc_dict = load_and_preprocess_data(args)
    elif args.data == 'mimic':
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, mc_idx_dict, num_mc_dict = load_and_preprocess_data_mimic(args)

    train_loader, val_loader, test_loader = create_loaders(data_dict, observed_idx_arr, labels, train_ids, valid_ids, test_ids, args.batch_size, args.num_workers, args.pin_memory, input_dims, transforms, masks, args.use_common_ids)
    num_experts_retriever = (args.num_candidates * num_modalities) + args.num_candidates_shared

    retriever_model = MoE_Retriever(num_modalities, args.num_patches, args.hidden_dim, num_experts_retriever, args.num_routers, args.top_k, args.num_heads, args.dropout).to(device)
    fusion_model = FusionLayer(num_modalities, args.num_patches, args.hidden_dim, n_labels, args.num_layers_fus, args.num_layers_pred, args.num_experts, args.num_routers, args.top_k, args.num_heads, args.dropout, args.fusion_sparse).to(device)
    params = list(fusion_model.parameters()) + [param for encoder in encoder_dict.values() for param in encoder.parameters()]    
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss() if args.data == 'adni' else torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device))
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    if args.data == 'adni':
        modality_dict = {'image':0, 'genomic': 1, 'clinical': 2, 'biospecimen': 3}
        char_to_modality = {'I': 'image', 'G': 'genomic', 'C': 'clinical', 'B': 'biospecimen'}

    elif args.data == 'mimic':
        modality_dict = {'lab':0, 'note': 1, 'code': 2}
        char_to_modality = {'L': 'lab', 'N': 'note', 'C': 'code'}


    mc_idx_dict_train = {k: list(set(v) & set(train_ids)) for k, v in mc_idx_dict.items()}
    mc_idx_dict_valid = {k: list(set(v) & set(train_ids+valid_ids)) for k, v in mc_idx_dict.items()}
    mc_idx_dict_test = {k: list(set(v) & set(train_ids+test_ids)) for k, v in mc_idx_dict.items()}


    for epoch in trange(args.train_epochs):
        if epoch < args.warm_up_epochs:
            moe_retriever=False
            fusion_model.train()

        else:
            moe_retriever=True
            if epoch == args.warm_up_epochs:
                optimizer.add_param_group({'params': retriever_model.parameters()})  # Add new parameters to existing optimizer
            fusion_model.train()
            retriever_model.train()

        for encoder in encoder_dict.values():
            encoder.train()
        
        task_losses = []
        gate_losses = []

        for batch_samples, batch_labels, batch_mcs, batch_observed in train_loader:
            batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            fusion_input = []
            for i, (modality, samples) in enumerate(batch_samples.items()):
                mask = batch_observed[:, modality_dict[modality]]
                encoded_samples = torch.zeros((samples.shape[0], args.num_patches, args.hidden_dim)).to(device)
                if mask.sum() > 0:
                    encoded_samples[mask] = encoder_dict[modality](samples[mask])
                if moe_retriever:
                    missing_sample_indices = (~mask).nonzero().flatten().cpu().numpy()
                    if len(missing_sample_indices) > 0:
                        ### MoE-Retriever ###
                        for missing_sample_idx in missing_sample_indices:
                            mc_num = batch_mcs[missing_sample_idx].item()
                            observed_modalities_char = num_mc_dict[mc_num]
                            observed_modalities = [char_to_modality[char] for char in observed_modalities_char]

                            inter_modal_length = len(observed_modalities) # args.num_supporting_samples 
                            supporting_group_indices = get_supporting_group(missing_modality=modality[0].upper(), observed_modalities=observed_modalities_char, mc_idx_dict=mc_idx_dict_train, num_samples=inter_modal_length)
                            input_arr = data_dict[modality][supporting_group_indices]
                            if modality == 'image':
                                input_arr = [process_2d_to_3d(data_dict[modality], idx, transforms=transforms['image'], masks=masks['image']) for idx in supporting_group_indices]
                                input_arr = np.stack(input_arr)
                            input_tensor = torch.tensor(input_arr, dtype=torch.float32).to(device)
                            intra_embeds = encoder_dict[modality](input_tensor)
                            inter_embeds = [encoder_dict[observed_modality](batch_samples[observed_modality][missing_sample_idx].unsqueeze(0)) for observed_modality in observed_modalities]
                            moe_retriever_input = torch.cat([intra_embeds, torch.cat(inter_embeds)])
                            
                            expert_idx_start = (args.modality).index(modality[0].upper())
                            expert_indices = list(range(expert_idx_start*args.num_candidates, (expert_idx_start+1)*args.num_candidates)) + list(range(num_experts_retriever-args.num_candidates_shared, num_experts_retriever))
                            encoded_samples[missing_sample_idx] = retriever_model(moe_retriever_input, expert_indices, num_intra=len(intra_embeds), num_inter=len(inter_embeds))
                        
                fusion_input.append(encoded_samples)

            outputs =  fusion_model(*fusion_input)
            
            task_loss = criterion(outputs, batch_labels)
            task_losses.append(task_loss.item())
            gate_loss = fusion_model.gate_loss()
            gate_losses.append(float(gate_loss))
            loss = task_loss + args.gate_loss_weight * gate_loss
            
            loss.backward()
            optimizer.step()
        
        if epoch < args.warm_up_epochs:
            print(f"[Seed {seed}/{args.n_runs-1}] [Warm-Up Epoch {epoch+1}/{args.warm_up_epochs}] Task Loss: {np.mean(task_losses):.2f}, Router Loss: {np.mean(gate_losses):.2f}")
        else:
            fusion_model.eval()
            retriever_model.eval()
            for encoder in encoder_dict.values():
                encoder.eval()
            
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for batch_samples, batch_labels, batch_mcs, batch_observed in val_loader:
                    batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
                    batch_labels = batch_labels.to(device, non_blocking=True)
                    batch_mcs = batch_mcs.to(device, non_blocking=True)
                    batch_observed = batch_observed.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    
                    fusion_input = []
                    for i, (modality, samples) in enumerate(batch_samples.items()):
                        mask = batch_observed[:, modality_dict[modality]]
                        encoded_samples = torch.zeros((samples.shape[0], args.num_patches, args.hidden_dim)).to(device)
                        if mask.sum() > 0:
                            encoded_samples[mask] = encoder_dict[modality](samples[mask])
                        if moe_retriever:
                            missing_sample_indices = (~mask).nonzero().flatten().cpu().numpy()
                            if len(missing_sample_indices) > 0:
                                ### MoE-Retriever ###
                                for missing_sample_idx in missing_sample_indices:
                                    mc_num = batch_mcs[missing_sample_idx].item()
                                    observed_modalities_char = num_mc_dict[mc_num]
                                    observed_modalities = [char_to_modality[char] for char in observed_modalities_char]
                            
                                    inter_modal_length = len(observed_modalities) 
                                    supporting_group_indices = get_supporting_group(missing_modality=modality[0].upper(), observed_modalities=observed_modalities_char, mc_idx_dict=mc_idx_dict_valid, num_samples=inter_modal_length)
                                    input_arr = data_dict[modality][supporting_group_indices]
                                    if modality == 'image':
                                        input_arr = [process_2d_to_3d(data_dict[modality], idx, transforms=transforms['image'], masks=masks['image']) for idx in supporting_group_indices]
                                        input_arr = np.stack(input_arr)
                                    input_tensor = torch.tensor(input_arr, dtype=torch.float32).to(device)
                                    intra_embeds = encoder_dict[modality](input_tensor)
                                    inter_embeds = [encoder_dict[observed_modality](batch_samples[observed_modality][missing_sample_idx].unsqueeze(0)) for observed_modality in observed_modalities]
                                    moe_retriever_input = torch.cat([intra_embeds, torch.cat(inter_embeds)])
                                    
                                    expert_idx_start = (args.modality).index(modality[0].upper())
                                    expert_indices = list(range(expert_idx_start*args.num_candidates, (expert_idx_start+1)*args.num_candidates)) + list(range(num_experts_retriever-args.num_candidates_shared, num_experts_retriever))
                                    encoded_samples[missing_sample_idx] = retriever_model(moe_retriever_input, expert_indices, num_intra=len(intra_embeds), num_inter=len(inter_embeds))
                                
                        fusion_input.append(encoded_samples)

                    outputs =  fusion_model(*fusion_input)

                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
                    if args.data == 'mimic':
                        all_probs.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                    else:
                        all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            val_auc = roc_auc_score(all_labels, all_probs) if args.data == 'mimic' else roc_auc_score(all_labels, all_probs, multi_class='ovr')

            print(f"[Seed {seed}/{args.n_runs-1}] [Epoch {epoch+1}/{args.train_epochs}] Task Loss: {np.mean(task_losses):.2f}, Router Loss: {np.mean(gate_losses):.2f} / Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}")

            if val_acc > best_val_acc:
                print(f" [(**Best**) Epoch {epoch+1}/{args.train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}")
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_val_auc = val_auc
                best_model_fus = deepcopy(fusion_model.state_dict())
                best_model_enc = {modality: deepcopy(encoder.state_dict()) for modality, encoder in encoder_dict.items()}
                best_model_retriever = deepcopy(retriever_model.state_dict())

                # Move the models to CPU for saving (only state_dict)
                if args.save:
                    best_model_fus_cpu = {k: v.cpu() for k, v in best_model_fus.items()}
                    best_model_enc_cpu = {modality: {k: v.cpu() for k, v in enc_state.items()} for modality, enc_state in best_model_enc.items()}
                    best_model_retriever_cpu = {k: v.cpu() for k, v in best_model_retriever.items()}

    # Save the best model
    if args.save:
        save_path = f'./saves/best_data_{args.data}_seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_acc_{best_val_acc*100:.2f}.pth'
        torch.save({
            'retreiver': best_model_retriever_cpu,
            'fusion_model': best_model_fus_cpu,
            'encoder_dict': best_model_enc_cpu
        }, save_path)

        print(f"Best model saved to {save_path}")
    
    # Load best model for test evaluation
    for modality, encoder in encoder_dict.items():
        encoder.load_state_dict(best_model_enc[modality])
        encoder.eval()
    retriever_model.load_state_dict(best_model_retriever)
    retriever_model.eval()
    fusion_model.load_state_dict(best_model_fus)
    fusion_model.eval()
        
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_samples, batch_labels, batch_mcs, batch_observed in test_loader:
            batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            fusion_input = []
            for i, (modality, samples) in enumerate(batch_samples.items()):
                mask = batch_observed[:, modality_dict[modality]]
                encoded_samples = torch.zeros((samples.shape[0], args.num_patches, args.hidden_dim)).to(device)
                if mask.sum() > 0:
                    encoded_samples[mask] = encoder_dict[modality](samples[mask])
                if moe_retriever:
                    missing_sample_indices = (~mask).nonzero().flatten().cpu().numpy()
                    if len(missing_sample_indices) > 0:
                        ### MoE-Retriever ###
                        for missing_sample_idx in missing_sample_indices:
                            mc_num = batch_mcs[missing_sample_idx].item()
                            observed_modalities_char = num_mc_dict[mc_num]
                            observed_modalities = [char_to_modality[char] for char in observed_modalities_char]
                            
                            inter_modal_length = len(observed_modalities) # args.num_supporting_samples
                            supporting_group_indices = get_supporting_group(missing_modality=modality[0].upper(), observed_modalities=observed_modalities_char, mc_idx_dict=mc_idx_dict_test, num_samples=inter_modal_length)
                            input_arr = data_dict[modality][supporting_group_indices]
                            if modality == 'image':
                                input_arr = [process_2d_to_3d(data_dict[modality], idx, transforms=transforms['image'], masks=masks['image']) for idx in supporting_group_indices]
                                input_arr = np.stack(input_arr)
                            input_tensor = torch.tensor(input_arr, dtype=torch.float32).to(device)
                            intra_embeds = encoder_dict[modality](input_tensor)
                            inter_embeds = [encoder_dict[observed_modality](batch_samples[observed_modality][missing_sample_idx].unsqueeze(0)) for observed_modality in observed_modalities]
                            moe_retriever_input = torch.cat([intra_embeds, torch.cat(inter_embeds)])
                            
                            expert_idx_start = (args.modality).index(modality[0].upper())
                            expert_indices = list(range(expert_idx_start*args.num_candidates, (expert_idx_start+1)*args.num_candidates)) + list(range(num_experts_retriever-args.num_candidates_shared, num_experts_retriever))
                            encoded_samples[missing_sample_idx] = retriever_model(moe_retriever_input, expert_indices, num_intra=len(intra_embeds), num_inter=len(inter_embeds))
                        
                fusion_input.append(encoded_samples)

            outputs =  fusion_model(*fusion_input)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            if args.data == 'mimic':
                all_probs.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            else:
                all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    test_auc = roc_auc_score(all_labels, all_probs) if args.data == 'mimic' else roc_auc_score(all_labels, all_probs, multi_class='ovr')

    return best_val_acc, best_val_f1, best_val_auc, test_acc, test_f1, test_auc

def main():
    args, _ = parse_args()
    logger = setup_logger('./logs', f'{args.data}', f'{args.modality}.txt')
    seeds = np.arange(args.n_runs) # [0, 1, 2]
    val_accs = []
    val_f1s = []
    val_aucs = []
    test_accs = []
    test_f1s = []
    test_aucs = []
    
    log_summary = "======================================================================================\n"
    
    model_kwargs = {
        "model": 'MoE-Retriever',
        "modality": args.modality,
        "initial_filling": args.initial_filling,
        "use_common_ids": args.use_common_ids,
        "train_epochs": args.train_epochs,
        "warm_up_epochs": args.warm_up_epochs,
        "num_candidates": args.num_candidates,
        "num_candidates_shared": args.num_candidates_shared,
        "num_supporting_samples": args.num_supporting_samples,
        "fusion_sparse": args.fusion_sparse,
        "num_experts": args.num_experts,
        "num_routers": args.num_routers,
        "top_k": args.top_k,
        "num_layers_enc": args.num_layers_enc,
        "num_layers_fus": args.num_layers_fus,
        "num_layers_pred": args.num_layers_pred,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "num_patches": args.num_patches,
        "gate_loss_weight": args.gate_loss_weight,
    }

    log_summary += f"Model configuration: {model_kwargs}\n"

    print('Modality:', args.modality)

    for seed in seeds:
        val_acc, val_f1, val_auc, test_acc, test_f1, test_auc = train_and_evaluate(args, seed)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_aucs.append(test_auc)
    
    val_avg_acc = np.mean(val_accs)*100
    val_std_acc = np.std(val_accs)*100
    val_avg_f1 = np.mean(val_f1s)*100
    val_std_f1 = np.std(val_f1s)*100
    val_avg_auc = np.mean(val_aucs)*100
    val_std_auc = np.std(val_aucs)*100

    test_avg_acc = np.mean(test_accs)*100
    test_std_acc = np.std(test_accs)*100
    test_avg_f1 = np.mean(test_f1s)*100
    test_std_f1 = np.std(test_f1s)*100
    test_avg_auc = np.mean(test_aucs)*100
    test_std_auc = np.std(test_aucs)*100

    log_summary += f'[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} '
    log_summary += f'[Val] Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} '
    log_summary += f'[Val] Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f} / '  
    log_summary += f'[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} '
    log_summary += f'[Test] Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} '
    log_summary += f'[Test] Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f} '  

    print(model_kwargs)
    print(f'[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} / Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f}')
    print(f'[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} / Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} / Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f}')

    logger.info(log_summary)

if __name__ == '__main__':
    main()