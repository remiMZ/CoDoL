#### CoOp

**Training Phase in single domain**

batch scripts/codol/main_ood_single_domain.sh pacs rn50_ood pacs_ood_art_photo 1 16 16

***Evaluation Phase in signe domain***
 
batch scripts/codol/eval_ood_single_domain.sh pacs rn50_ood pacs_ood_art_photo 1 16 16





**Training Phase in multi domain**

batch scripts/codol/main_ood_multi_domain.sh pacs rn50_ood pacs_ood_art_painting 1 16 16

***Evaluation Phase in signe domain***
 
batch scripts/codol/eval_ood_multi_domain.sh pacs rn50_ood pacs_ood_art_painting 1 16 16