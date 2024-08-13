import os
# configs_list = [
#     './configs/cfg_voc21.py',
#     './configs/cfg_context60.py',
#     './configs/cfg_coco_object.py',
# ]
configs_list = [
    'voc21',
    'context60',
    'coco_object',
]


for config in configs_list:
    print(f"Running {config}")
    # os.system(f"bash ./dist_test.sh {config} --")
    
    os.system(f"python eval_sclip.py --config ./configs/cfg_{config}.py  --work-dir ./work_logs_sclip_{config}/")
    

