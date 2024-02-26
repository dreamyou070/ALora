import abc
class AttentionStore :

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.heatmap_store = {}
        self.self_query_store = {}
        self.self_key_store = {}
        self.self_value_store = {}
        self.cross_query_store = {}
        self.cross_key_store = {}
        self.cross_value_store = {}
        self.query_dict = {}
        self.key_dict = {}
        self.value_dict = {}
        self.repeat = 0
        self.normal_score_list = []
        self.map_dict = {}
        self.query_dict_sub = {}
        self.batchshaped_query_dict = {}
        self.batchshaped_key_dict = {}
        self.scale_dict = {}

    def get_empty_store(self):
        return {}

    def save_query(self, query, layer_name):
        if layer_name not in self.query_dict.keys():
            self.query_dict[layer_name] = []
            self.query_dict[layer_name].append(query)
        else:
            self.query_dict[layer_name].append(query)

    def save_map(self, map, layer_name):
        if layer_name not in self.map_dict.keys():
            self.map_dict[layer_name] = []
            self.map_dict[layer_name].append(map)
        else:
            self.map_dict[layer_name].append(map)

    def save_batshaped_qk(self, query, key, layer_name):
        if layer_name not in self.batchshaped_query_dict.keys():
            self.batchshaped_query_dict[layer_name] = []
            self.batchshaped_key_dict[layer_name] = []
        self.batchshaped_query_dict[layer_name].append(query)
        self.batchshaped_key_dict[layer_name].append(key)


    def save_scale(self, scale, layer_name):
        if layer_name not in self.scale_dict.keys():
            self.scale_dict[layer_name] = []
        self.scale_dict[layer_name].append(scale)

    def store_classifocation_map(self, map, layer_name):
        if layer_name not in self.classification_map_dict.keys():
            self.classification_map_dict[layer_name] = []
            self.classification_map_dict[layer_name].append(map)
        else:
            self.classification_map_dict[layer_name].append(map)


    def store_normal_score(self, score):
        self.normal_score_list.append(score)

    def save_query_sub(self, query, layer_name):
        if layer_name not in self.query_dict_sub.keys():
            self.query_dict_sub[layer_name] = []
            self.query_dict_sub[layer_name].append(query)
        else:
            self.query_dict_sub[layer_name].append(query)

    def store(self, attn, layer_name):
        if layer_name not in self.step_store.keys() :
            self.step_store[layer_name] = []
            self.step_store[layer_name].append(attn)
        else :
            self.step_store[layer_name].append(attn)
            #self.step_store[layer_name] = self.step_store[layer_name] + attn
        return attn




    def self_query_key_value_caching(self,query_value, key_value, value_value, layer_name):

        if layer_name not in self.self_query_store.keys() :
            self.self_query_store[layer_name] = []
            self.self_key_store[layer_name] = []
            self.self_value_store[layer_name] = []
            self.self_query_store[layer_name].append(query_value)
            self.self_key_store[layer_name].append(key_value)
            self.self_value_store[layer_name].append(value_value)
        else :
            self.self_query_store[layer_name].append(query_value)
            self.self_key_store[layer_name].append(key_value)
            self.self_value_store[layer_name].append(value_value)

    def save(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store[key].append(attn.clone())
        return attn

    def between_steps(self):
        assert len(self.attention_store) == 0

        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.heatmap_store = {}
        self.self_query_store = {}
        self.self_key_store = {}
        self.self_value_store = {}
        self.cross_query_store = {}
        self.cross_key_store = {}
        self.cross_value_store = {}
        self.query_dict = {}
        self.key_dict = {}
        self.value_dict = {}
        self.repeat = 0
        self.normal_score_list = []
        self.map_dict = {}
        self.query_dict_sub = {}
        self.batchshaped_query_dict = {}
        self.batchshaped_key_dict = {}
        self.scale_dict = {}


# layer_name : down_blocks_0_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.15811388300841897
# layer_name : down_blocks_0_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.15811388300841897
# layer_name : down_blocks_0_attentions_1_transformer_blocks_0_attn1
# alpha scale : 0.15811388300841897
# layer_name : down_blocks_0_attentions_1_transformer_blocks_0_attn2
# alpha scale : 0.15811388300841897
# layer_name : down_blocks_1_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.11180339887498948
# layer_name : down_blocks_1_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.11180339887498948
# layer_name : down_blocks_1_attentions_1_transformer_blocks_0_attn1
# alpha scale : 0.11180339887498948
# layer_name : down_blocks_1_attentions_1_transformer_blocks_0_attn2
# alpha scale : 0.11180339887498948
# layer_name : down_blocks_2_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.07905694150420949
# layer_name : down_blocks_2_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.07905694150420949
# layer_name : down_blocks_2_attentions_1_transformer_blocks_0_attn1
# alpha scale : 0.07905694150420949
# layer_name : down_blocks_2_attentions_1_transformer_blocks_0_attn2
# alpha scale : 0.07905694150420949
# layer_name : mid_block_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.07905694150420949
# layer_name : mid_block_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_1_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_1_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_1_attentions_1_transformer_blocks_0_attn1
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_1_attentions_1_transformer_blocks_0_attn2
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_1_attentions_2_transformer_blocks_0_attn1
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_1_attentions_2_transformer_blocks_0_attn2
# alpha scale : 0.07905694150420949
# layer_name : up_blocks_2_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.11180339887498948
# layer_name : up_blocks_2_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.11180339887498948
# layer_name : up_blocks_2_attentions_1_transformer_blocks_0_attn1
# alpha scale : 0.11180339887498948
# layer_name : up_blocks_2_attentions_1_transformer_blocks_0_attn2
# alpha scale : 0.11180339887498948
# layer_name : up_blocks_2_attentions_2_transformer_blocks_0_attn1
# alpha scale : 0.11180339887498948
# layer_name : up_blocks_2_attentions_2_transformer_blocks_0_attn2
# alpha scale : 0.11180339887498948
# layer_name : up_blocks_3_attentions_0_transformer_blocks_0_attn1
# alpha scale : 0.15811388300841897
# layer_name : up_blocks_3_attentions_0_transformer_blocks_0_attn2
# alpha scale : 0.15811388300841897
# layer_name : up_blocks_3_attentions_1_transformer_blocks_0_attn1
# alpha scale : 0.15811388300841897
# layer_name : up_blocks_3_attentions_1_transformer_blocks_0_attn2
# alpha scale : 0.15811388300841897
# layer_name : up_blocks_3_attentions_2_transformer_blocks_0_attn1
# alpha scale : 0.15811388300841897
# layer_name : up_blocks_3_attentions_2_transformer_blocks_0_attn2
# alpha scale : 0.15811388300841897