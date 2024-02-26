import torch
import torch.nn as nn

def passing_normalize_argument(args) :
    global argument
    argument = args

class NormalActivator(nn.Module):

    def __init__(self, loss_focal, loss_l2, use_focal_loss):
        super(NormalActivator, self).__init__()

        # [1]
        self.anomal_feat_list = []
        self.normal_feat_list = []
        # [2]
        self.attention_loss = {}
        self.attention_loss['normal_cls_loss'] = []
        self.attention_loss['anomal_cls_loss'] = []
        self.attention_loss['normal_trigger_loss'] = []
        self.attention_loss['anomal_trigger_loss'] = []
        self.trigger_score = []
        self.cls_score = []
        # [3]
        self.loss_focal = loss_focal
        self.loss_l2 = loss_l2
        self.anomal_map_loss = []
        self.use_focal_loss = use_focal_loss
        # [4]
        self.normal_matching_query_loss = []
        self.resized_queries = []
        self.queries = []
        self.resized_attn_scores = []
        self.noise_prediction_loss = []

    def collect_queries(self, origin_query, normal_position, anomal_position, do_collect_normal):

        pix_num = origin_query.shape[0]
        for pix_idx in range(pix_num):
            feat = origin_query[pix_idx].squeeze(0)
            normal_flag = normal_position[pix_idx]
            anomal_flag = anomal_position[pix_idx]
            if normal_flag == 1:
                if do_collect_normal:
                    self.normal_feat_list.append(feat.unsqueeze(0))

            elif anomal_flag == 1 :
                self.anomal_feat_list.append(feat.unsqueeze(0))



    def collect_queries_normal(self, origin_query, normal_position_vector, do_collect_normal):
        # check foreground normal collecting code
        pix_num = origin_query.shape[0]
        for pix_idx in range(pix_num):
            feat = origin_query[pix_idx].squeeze(0)
            normal_flag = normal_position_vector[pix_idx]
            if normal_flag == 1 and do_collect_normal :
                self.normal_feat_list.append(feat.unsqueeze(0))
         

    def collect_attention_scores(self, attn_score, anomal_position_vector,
                                 do_normal_activating = True):


        # [1] preprocessing
        cls_score, trigger_score = attn_score.chunk(2, dim=-1)
        cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()      # head, pix_num

        cls_score, trigger_score = cls_score.mean(dim=0), trigger_score.mean(dim=0)  # pix_num
        total_score = torch.ones_like(cls_score)

        # [2]
        normal_cls_score = cls_score * (1 - anomal_position_vector)
        normal_trigger_score = trigger_score * (1 - anomal_position_vector)
        anomal_cls_score = cls_score * anomal_position_vector
        anomal_trigger_score = trigger_score * anomal_position_vector


        # [3]
        normal_cls_score = normal_cls_score / total_score
        normal_trigger_score = normal_trigger_score / total_score
        anomal_cls_score = anomal_cls_score / total_score
        anomal_trigger_score = anomal_trigger_score / total_score

        # [4]
        normal_cls_loss = normal_cls_score ** 2
        normal_trigger_loss = (1 - normal_trigger_score ** 2 )  #normal cls score 이랑 같은 상황
        anomal_cls_loss = (1 - anomal_cls_score ** 2)
        anomal_trigger_loss = anomal_trigger_score ** 2

        # [5]
        if do_normal_activating:
            self.attention_loss['normal_cls_loss'].append(normal_cls_loss.mean())
            self.attention_loss['normal_trigger_loss'].append(normal_trigger_loss.mean())

        anomal_pixel_num = anomal_position_vector.sum()
        if anomal_pixel_num > 0:
            self.attention_loss['anomal_cls_loss'].append(anomal_cls_loss.mean())
            self.attention_loss['anomal_trigger_loss'].append(anomal_trigger_loss.mean())

    def collect_anomal_map_loss(self, attn_score, anomal_position_vector):

        if self.use_focal_loss:

            cls_score, trigger_score = attn_score.chunk(2, dim=-1)
            cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()  # head, pix_num
            cls_score, trigger_score = cls_score.mean(dim=0), trigger_score.mean(dim=0)  # pix_num
            res = int(attn_score.shape[0] ** 0.5)
            focal_loss_in = torch.cat([cls_score.view(res, res).unsqueeze(0).unsqueeze(0),
                                       trigger_score.view(res, res).unsqueeze(0).unsqueeze(0)], 1)

            # [2] target
            focal_loss_trg = anomal_position_vector.view(res, res).unsqueeze(0).unsqueeze(0)
            map_loss = self.loss_focal(focal_loss_in,
                                       focal_loss_trg.to(dtype=trigger_score.dtype))

        else:
            cls_score, trigger_score = attn_score.chunk(2, dim=-1)
            cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()  # head, pix_num
            cls_score, trigger_score = cls_score.mean(dim=0), trigger_score.mean(dim=0)  # pix_num
            trg_trigger_score = 1 - anomal_position_vector
            map_loss = self.loss_l2(trigger_score.float(), trg_trigger_score.float())

        self.anomal_map_loss.append(map_loss)

    def collect_noise_prediction_loss(self, noise_pred, noise, anomal_position_vector):    
        b, c, h, w = noise_pred.shape
        anomal_position = anomal_position_vector.reshape(h, w).unsqueeze(0).unsqueeze(0)
        anomal_position = anomal_position.repeat(b, c, 1, 1)
        trg_noise_pred = noise * anomal_position # only anomal is noise, normal is zero (not noise)
        noise_pred_loss = self.loss_l2(noise_pred.float(), trg_noise_pred.float())
        self.noise_prediction_loss.append(noise_pred_loss)
        
        
    def generate_mahalanobis_distance_loss(self):

        def mahal(u, v, cov):
            delta = u - v
            m = torch.dot(delta, torch.matmul(cov, delta))
            return torch.sqrt(m)


        normal_feats = torch.cat(self.normal_feat_list, dim=0)
        if argument.mahalanobis_normalize :
            """ add noramlizing """
            normal_feats = torch.nn.functional.normalize(normal_feats, p=2, dim=1, eps=1e-12, out=None)

        mu = torch.mean(normal_feats, dim=0)
        cov = torch.cov(normal_feats.transpose(0, 1))
        normal_mahalanobis_dists = [mahal(feat, mu, cov) for feat in normal_feats]
        normal_dist_max = torch.tensor(normal_mahalanobis_dists).max()
        normal_dist_mean = torch.tensor(normal_mahalanobis_dists).mean()

        if len(self.anomal_feat_list) > 0:
            anomal_feats = torch.cat(self.anomal_feat_list, dim=0)
            anomal_mahalanobis_dists = [mahal(feat, mu, cov) for feat in anomal_feats]
            anomal_dist_mean = torch.tensor(anomal_mahalanobis_dists).mean()
            total_dist = normal_dist_mean + anomal_dist_mean
            normal_dist_loss = (normal_dist_mean / total_dist).requires_grad_()
            normal_dist_max = normal_dist_max.requires_grad_()
        else:
            normal_dist_loss = normal_dist_max.requires_grad_()
        self.normal_feat_list = []
        self.anomal_feat_list = []

        return normal_dist_loss, normal_dist_mean, normal_dist_max

    def generate_attention_loss(self):

        normal_cls_loss = 0.0
        normal_trigger_loss = 0.0
        if len(self.attention_loss['normal_cls_loss']) != 0:
            normal_cls_loss = torch.stack(self.attention_loss['normal_cls_loss'], dim=0).mean(dim=0)
            normal_trigger_loss = torch.stack(self.attention_loss['normal_trigger_loss'], dim=0).mean(dim=0)

        anomal_cls_loss = 0.0
        anomal_trigger_loss = 0.0
        if len(self.attention_loss['anomal_cls_loss']) != 0:
            anomal_cls_loss = torch.stack(self.attention_loss['anomal_cls_loss'], dim=0).mean(dim=0)
            anomal_trigger_loss = torch.stack(self.attention_loss['anomal_trigger_loss'], dim=0).mean(dim=0)

        self.attention_loss = {'normal_cls_loss': [], 'normal_trigger_loss': [],
                               'anomal_cls_loss': [], 'anomal_trigger_loss': []}
        return normal_cls_loss, normal_trigger_loss, anomal_cls_loss, anomal_trigger_loss

    def generate_anomal_map_loss(self):
        map_loss = torch.stack(self.anomal_map_loss, dim=0)
        map_loss = map_loss.mean()
        self.anomal_map_loss = []
        return map_loss
    
    def generate_noise_prediction_loss(self):
        noise_pred_loss = torch.stack(self.noise_prediction_loss, dim=0)
        noise_pred_loss = noise_pred_loss.mean()
        self.noise_prediction_loss = []
        return noise_pred_loss

        

    def resize_query_features(self, query) :

        if query.dim() == 2:
            query = query.unsqueeze(0)
        head_num, pix_num, dim = query.shape
        res = int(pix_num ** 0.5)
        query_map = query.view(head_num, res, res, dim).permute(0, 3, 1, 2).contiguous() # batch, channel, res, res
        resized_query_map = nn.functional.interpolate(query_map, size=(64, 64), mode='bilinear')
        resized_query = resized_query_map.permute(0, 2, 3, 1).contiguous().view(head_num, -1, dim).squeeze()
        self.resized_queries.append(resized_query) # len = 3


    def resize_attn_scores(self, attn_score) :
        # attn_score = [head, pix_num, sen_len]
        head_num, pix_num, sen_len = attn_score.shape
        res = int(pix_num ** 0.5)
        attn_map = attn_score.view(head_num, res, res, sen_len).permute(0, 3, 1, 2).contiguous()
        resized_attn_map = nn.functional.interpolate(attn_map, size=(64, 64), mode='bilinear')
        resized_attn_score = resized_attn_map.permute(0, 2, 3, 1).contiguous().view(head_num, -1, sen_len)  # 8, 64*64, sen_len
        self.resized_attn_scores.append(resized_attn_score) # len = 3

    def generate_conjugated(self,):
        concat_query = torch.cat(self.resized_queries, dim=-1).squeeze()     # 4096, 1960 ***
        self.resized_queries = []
        return concat_query

    def generate_conjugated_attn_score(self,):
        concat_attn_score = torch.cat(self.resized_attn_scores, dim=0)     # 8, 4096, sen_len ***
        self.resized_attn_scores = []
        return concat_attn_score[:,:,:2]

    def reset(self) -> None:

        # [1]
        self.anomal_feat_list = []
        self.normal_feat_list = []
        # [2]
        self.attention_loss = {'normal_cls_loss': [], 'normal_trigger_loss': [],
                               'anomal_cls_loss': [], 'anomal_trigger_loss': []}
        self.trigger_score = []
        self.cls_score = []
        # [3]
        self.anomal_map_loss = []
        # [4]
        self.normal_matching_query_loss = []
        self.resized_queries = []
        self.queries = []
        self.resized_attn_scores = []
        self.noise_prediction_loss = []