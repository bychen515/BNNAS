from config import config
import pickle

op_flops_dict = pickle.load(open(config.flops_lookup_table, 'rb'))

cand = [3, -1, 4, 4, 4, -1, -1, 0, 0, -1, 4, 0, 4, 0, 0, 2, 4, 2, 2, 4, 2]

preprocessing_flops = op_flops_dict['PreProcessing'][config.backbone_info[0]]
postprocessing_flops = op_flops_dict['PostProcessing'][config.backbone_info[-1]]
total_flops = preprocessing_flops + postprocessing_flops
for i in range(len(cand)):
    inp, oup, img_h, img_w, stride = config.backbone_info[i + 1]
    op_id = cand[i]
    if op_id >= 0:
        key = config.blocks_keys[op_id]
        total_flops += op_flops_dict[key][(inp, oup, img_h, img_w, stride)]
print(total_flops)