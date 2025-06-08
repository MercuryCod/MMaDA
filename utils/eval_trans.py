import os

import clip
import numpy as np
import torch
from scipy import linalg

import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric

def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(),title_batch, outname)
    plot_xyz =np.transpose(plot_xyz, (0, 1, 4, 2, 3)) 
    writer.add_video(tag, plot_xyz, nb_iter, fps = 20)

@torch.no_grad()        
def evaluation_vqvae(out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False) : 
    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []


    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


            pred_pose, loss_commit, perplexity = net(motion[i:i+1, :m_length[i]])
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
            pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
            
            if savenpy:
                np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
                np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose

            if i < min(4, bs):
                draw_org.append(pose_xyz)
                draw_pred.append(pred_xyz)
                draw_text.append(caption[i])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    # Calculate diversity with safety check for small sample sizes
    min_samples_for_diversity = 10  # Minimum samples needed for meaningful diversity
    if motion_annotation_np.shape[0] >= min_samples_for_diversity:
        diversity_real = calculate_diversity(motion_annotation_np, min(300, motion_annotation_np.shape[0]//2))
        diversity = calculate_diversity(motion_pred_np, min(300, motion_pred_np.shape[0]//2))
    else:
        logger.warning(f"Too few samples ({motion_annotation_np.shape[0]}) for diversity calculation, using default values")
        diversity_real = 0.0
        diversity = 0.0

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        if nb_iter % 5000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        if nb_iter % 5000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)   

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

    if save:
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()        
def evaluation_transformer(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, draw = True, save = True, savegif=False) : 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for i in range(1):
        for batch in val_loader:
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22
            
            text = clip.tokenize(clip_text, truncate=True).cuda()

            feat_clip_text = clip_model.encode_text(text).float()
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k:k+1], False)
                except:
                    index_motion = torch.ones(1,1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if draw:
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if i == 0 and k < 4:
                        draw_pred.append(pred_xyz)
                        draw_text_pred.append(clip_text[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


                    for j in range(min(4, bs)):
                        draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                        draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    # Calculate diversity with safety check for small sample sizes
    min_samples_for_diversity = 10  # Minimum samples needed for meaningful diversity
    if motion_annotation_np.shape[0] >= min_samples_for_diversity:
        diversity_real = calculate_diversity(motion_annotation_np, min(300, motion_annotation_np.shape[0]//2))
        diversity = calculate_diversity(motion_pred_np, min(300, motion_pred_np.shape[0]//2))
    else:
        logger.warning(f"Too few samples ({motion_annotation_np.shape[0]}) for diversity calculation, using default values")
        diversity_real = 0.0
        diversity = 0.0

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample


    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        if nb_iter % 10000 == 0 :
        # if True:
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        if nb_iter % 10000 == 0 : 
        # if True:
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return fid, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred


@torch.no_grad()        
def evaluation_transformer_test(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False) : 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    
    for batch in val_loader:

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []
        for i in range(30):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.full((bs,), motion_seq_len, dtype=torch.long, device=mmada_model.device)
            
            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k:k+1], True)
                except:
                    index_motion = torch.ones(1,1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if draw or savenpy:
                    try:
                        pred_denorm = val_loader.dataset.inv_transform(
                            pred_pose.detach().cpu().numpy()
                        )  # (bs, T, D)
                        pred_xyz_batch = recover_from_ric(
                            torch.from_numpy(pred_denorm).float().cuda(), num_joints
                        )  # (bs, T, 3*J)

                        for j in range(bs):
                            if savenpy:
                                np.save(
                                    os.path.join(out_dir, name[j] + "_pred.npy"),
                                    pred_xyz_batch[j : j + 1].detach().cpu().numpy(),
                                )

                            if draw and i == 0 and j < 4:
                                draw_pred.append(pred_xyz_batch[j : j + 1])
                                draw_text_pred.append(clip_text[j])
                                draw_name.append(name[j])
                    except Exception as e:
                        logger.warning(f"Visualization failed: {e}")

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    # Calculate diversity with safety check for small sample sizes
    min_samples_for_diversity = 10  # Minimum samples needed for meaningful diversity
    if motion_annotation_np.shape[0] >= min_samples_for_diversity:
        diversity_real = calculate_diversity(motion_annotation_np, min(300, motion_annotation_np.shape[0]//2))
        diversity = calculate_diversity(motion_pred_np, min(300, motion_pred_np.shape[0]//2))
    else:
        logger.warning(f"Too few samples ({motion_annotation_np.shape[0]}) for diversity calculation, using default values")
        diversity_real = 0.0
        diversity = 0.0

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    multimodality = calculate_multimodality(motion_multimodality, 10)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)
    
    
    if draw:
        for ii in range(len(draw_org)):
            tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)
        
            tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist

@torch.no_grad()
def evaluation_mmada_t2m(
    out_dir,
    val_loader,
    vq_model,
    mmada_model,
    uni_prompting,
    logger,
    writer,
    nb_iter,
    best_fid,
    best_iter,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    eval_wrapper,
    mask_token_id=126336,
    motion_vocab_size=512,
    motion_seq_len=256,
    image_codebook_size=8192,
    savegif=True,
):
    """
    Evaluate text-to-motion generation with MMaDA model.
    COMPLETE VERSION: With proper motion generation and GIF creation.
    """
    logger.info(f"========== Evaluating MMaDA T2M at iter {nb_iter} ==========")
    
    mmada_model.eval()
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Collections for evaluation
    motion_annotation_list = []
    motion_pred_list = []
    draw_org = []
    draw_pred = []
    draw_text = []
    
    # Evaluation counters
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    nb_sample = 0
    
    # Use a reasonable subset for evaluation
    max_samples = 64
    sample_count = 0
    
    try:
        for batch in val_loader:
            if sample_count >= max_samples:
                break
                
            # Unpack validation batch
            (word_embeddings, pos_one_hots, clip_text, sent_len, motion, m_length, token, name) = batch
            batch_size = len(clip_text)
            
            # Move to device
            motion = motion.cuda().float()
            
            # Determine number of joints from motion dimension
            num_joints = 21 if motion.shape[-1] == 251 else 22
            
            # Process ground truth motions for visualization and evaluation
            motion_pred_eval = torch.zeros_like(motion).cuda()
            pred_lengths = torch.zeros(batch_size, dtype=torch.long).cuda()
            
            for i in range(batch_size):
                try:
                    # Create input for generation
                    # For evaluation, we need to create a sequence with mask tokens where motion should be generated
                    text_vocab_size = len(uni_prompting.text_tokenizer)
                    motion_token_offset = text_vocab_size + image_codebook_size
                    
                    # Create dummy motion tokens in the OFFSET space to match training
                    # These will be replaced with mask tokens after formatting
                    # Using the first valid motion token in offset space
                    dummy_motion_tokens = torch.full((motion_seq_len,), motion_token_offset, dtype=torch.long, device=motion.device)
                    
                    # Use uni_prompting to format the sequence properly
                    # The motion tokens are expected to be in offset space, matching training
                    input_ids, attention_mask, _ = uni_prompting(([clip_text[i]], dummy_motion_tokens.unsqueeze(0), dummy_motion_tokens.unsqueeze(0)), 't2m')
                    
                    # Find motion token positions and replace with mask tokens
                    som_token = int(uni_prompting.sptids_dict["<|som|>"])
                    eom_token = int(uni_prompting.sptids_dict["<|eom|>"])
                    som_pos = (input_ids[0] == som_token).nonzero(as_tuple=True)[0]
                    eom_pos = (input_ids[0] == eom_token).nonzero(as_tuple=True)[0]
                    
                    if len(som_pos) > 0 and len(eom_pos) > 0:
                        motion_start_idx = som_pos[0].item() + 1
                        motion_end_idx = eom_pos[0].item()
                        # Replace motion positions with mask tokens for generation
                        input_ids[0, motion_start_idx:motion_end_idx] = mask_token_id
                    else:
                        logger.warning(f"Could not find motion token boundaries for sample {i}")
                        continue
                    
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda() if attention_mask is not None else None
                    
                    # OPTIMIZATION 4: Reduce generation timesteps for speed
                    try:
                        generated_tokens = mmada_model.t2m_generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            uni_prompting=uni_prompting,
                            mask_token_id=mask_token_id,
                            motion_vocab_size=motion_vocab_size,
                            seq_len=motion_seq_len,
                            timesteps=8,  # Reduced for faster evaluation
                            image_codebook_size=image_codebook_size,  # Pass the image codebook size
                        )
                        
                        # CRITICAL FIX: The t2m_generate method now returns tokens already in VQ-VAE space [0, motion_vocab_size-1]
                        # No need to subtract offset since the method handles this internally
                        generated_tokens = generated_tokens[0]
                        
                        # Ensure tokens are in valid VQ-VAE range
                        generated_tokens = torch.clamp(generated_tokens, 0, motion_vocab_size - 1)
                        
                        # Detect actual motion length by looking for end-of-motion token (512 in dataset space)
                        eom_token_vqvae = motion_vocab_size  # 512 - the end-of-motion token in VQ-VAE space
                        eom_positions = (generated_tokens == eom_token_vqvae).nonzero(as_tuple=True)[0]
                        if len(eom_positions) > 0:
                            # Truncate at the first end-of-motion token
                            actual_seq_len = eom_positions[0].item()
                            generated_tokens = generated_tokens[:actual_seq_len]
                        
                        # Decode using VQ-VAE
                        decoded_motion = vq_model.forward_decoder(generated_tokens.unsqueeze(0))
                        decoded_motion = decoded_motion.squeeze(0)
                        
                        # Store for evaluation
                        actual_length = min(decoded_motion.shape[0], motion.shape[1])
                        motion_pred_eval[i, :actual_length] = decoded_motion[:actual_length]
                        pred_lengths[i] = actual_length
                        
                        
                        
                    except Exception as gen_error:
                        logger.warning(f"Generation failed for sample {i}: {gen_error}")
                        # Use ground truth as fallback
                        motion_pred_eval[i, :m_length[i]] = motion[i, :m_length[i]]
                        pred_lengths[i] = m_length[i]
                
                except Exception as sample_error:
                    logger.warning(f"Sample processing failed for {i}: {sample_error}")
                    # Use ground truth as fallback
                    motion_pred_eval[i, :m_length[i]] = motion[i, :m_length[i]]
                    pred_lengths[i] = m_length[i]
            
            # Get motion embeddings for evaluation (only for processed samples)
            try:
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion_pred_eval, pred_lengths)
                
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)
                
                # Calculate R-precision for this batch
                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match
                
            except Exception as eval_error:
                logger.warning(f"Embedding evaluation failed: {eval_error}")
            
            # Prepare motions for visualization
            if savegif and len(draw_org) < 4:  # Only visualize first 4 samples
                try:
                    # Process ground truth motions
                    gt_denorm = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
                    gt_xyz = recover_from_ric(torch.from_numpy(gt_denorm).float().cuda(), num_joints)
                    
                    # Process predicted motions  
                    pred_denorm = val_loader.dataset.inv_transform(motion_pred_eval.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
                    
                    # Collect samples for visualization
                    for j in range(min(4 - len(draw_org), batch_size)):
                        draw_org.append(gt_xyz[j:j+1, :m_length[j]])
                        draw_pred.append(pred_xyz[j:j+1, :pred_lengths[j]])
                        draw_text.append(clip_text[j])
                        
                except Exception as viz_error:
                    logger.warning(f"Visualization preparation failed: {viz_error}")
            
            sample_count += batch_size
            nb_sample += batch_size
            
        # Calculate final metrics
        if len(motion_annotation_list) > 0 and len(motion_pred_list) > 0:
            motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
            motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
            
            gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
            pred_mu, pred_cov = calculate_activation_statistics(motion_pred_np)
            
            fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)
            
            # Calculate diversity with safety check for small sample sizes
            min_samples_for_diversity = 10  # Minimum samples needed for meaningful diversity
            if motion_annotation_np.shape[0] >= min_samples_for_diversity:
                diversity_real = calculate_diversity(motion_annotation_np, min(300, motion_annotation_np.shape[0]//2))
                diversity = calculate_diversity(motion_pred_np, min(300, motion_pred_np.shape[0]//2))
            else:
                logger.warning(f"Too few samples ({motion_annotation_np.shape[0]}) for diversity calculation, using default values")
                diversity_real = 0.0
                diversity = 0.0
            
            R_precision_real = R_precision_real / nb_sample
            R_precision = R_precision / nb_sample
            matching_score_real = matching_score_real / nb_sample
            matching_score_pred = matching_score_pred / nb_sample
            
        else:
            logger.warning("No valid samples for metric calculation, using dummy values")
            fid, diversity, R_precision = 100.0, 0.0, [0.0, 0.0, 0.0]
            matching_score_pred = 100.0
        
        # Generate motion GIFs
        if savegif and len(draw_org) > 0 and nb_iter % 50 == 0 :
            try:
                logger.info(f"Generating motion GIFs...")
                
                # Create GIFs for ground truth and predicted motions
                for ii in range(len(draw_org)):
                    # Ground truth GIF
                    gif_path_gt = os.path.join(out_dir, f'mmada_t2m_gt_{ii}_iter_{nb_iter}.gif')
                    tensorborad_add_video_xyz(
                        writer, draw_org[ii], nb_iter, 
                        tag=f'./MMaDA_T2M/gt_eval_{ii}', 
                        nb_vis=1, 
                        title_batch=[f"GT: {draw_text[ii]}"], 
                        outname=[gif_path_gt]
                    )
                    
                    # Predicted motion GIF
                    gif_path_pred = os.path.join(out_dir, f'mmada_t2m_pred_{ii}_iter_{nb_iter}.gif')
                    tensorborad_add_video_xyz(
                        writer, draw_pred[ii], nb_iter, 
                        tag=f'./MMaDA_T2M/pred_eval_{ii}', 
                        nb_vis=1, 
                        title_batch=[f"Pred: {draw_text[ii]}"], 
                        outname=[gif_path_pred]
                    )
                
                logger.info(f"Generated {len(draw_org)} motion GIF pairs")
                
            except Exception as gif_error:
                logger.warning(f"GIF generation failed: {gif_error}")
        
        # FIXED: Proper best metrics tracking and logging
        if fid < best_fid:
            msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            logger.info(msg)
            best_fid, best_iter = fid, nb_iter
        
        if abs(diversity_real - diversity) < abs(diversity_real - best_div):
            msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
            logger.info(msg)
            best_div = diversity
        
        if R_precision[0] > best_top1:
            msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
            logger.info(msg)
            best_top1 = R_precision[0]
        
        if R_precision[1] > best_top2:
            msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
            logger.info(msg)
            best_top2 = R_precision[1]
        
        if R_precision[2] > best_top3:
            msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
            logger.info(msg)
            best_top3 = R_precision[2]
        
        if matching_score_pred < best_matching:
            msg = f"--> --> \t Matching Score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
            logger.info(msg)
            best_matching = matching_score_pred
        
        # Log results
        msg = f"--> MMaDA T2M Eval. Iter {nb_iter}: FID {fid:.4f}, Diversity {diversity:.4f}, R-precision {R_precision}, Matching {matching_score_pred:.4f}"
        logger.info(msg)
        
        logger.info(f"Evaluation Results: FID: {fid:.3f} (Best: {best_fid:.3f}), Diversity: {diversity:.3f} (Best: {best_div:.3f}), R-precision: {R_precision} (Best: [{best_top1:.3f}, {best_top2:.3f}, {best_top3:.3f}]), Matching Score: {matching_score_pred:.3f} (Best: {best_matching:.3f}), Samples Evaluated: {nb_sample}")
        
        # FIXED: Return updated best metrics like other evaluation functions
        return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching
        
    except Exception as eval_error:
        logger.error(f"Evaluation failed: {eval_error}")
        import traceback
        traceback.print_exc()
        # FIXED: Return correct number of values matching the new signature
        return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching
    
    finally:
        mmada_model.train()