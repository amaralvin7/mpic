import src.tools as tools
import sys

pairs = (('splits_hitloop_B.json', 'targetRR_top1k.json'),
         ('splits_hitloop_C.json', 'targetRR_verify.json'),
         ('splits_hitloop_D.json', 'targetRR_minboost.json'))

for o, n in pairs:
    od = tools.load_json(f'../data/{o}')
    nd = tools.load_json(f'../data/{n}')
    for s in ('train', 'val'):
        all_o_fps = []
        for d in od.keys():
            all_o_fps.extend(od[d][s])
        comp = sorted(all_o_fps) == sorted(nd[s])
        print(s, comp)
        # for i, _ in enumerate(all_o_fps):
        #     print(all_o_fps[i], nd[s][i])
        #     if all_o_fps[i] != nd[s][i]:
        #         sys.exit()
        # comp = all_o_fps == nd[s]
        # print(s, comp)
