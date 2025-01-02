python utools/feature_encoder.py --dataset HIV --gpu 1
python utools/feature_encoder.py --dataset SAbDab --gpu 1 --use_cdr
python utools/feature_encoder.py --dataset AVIDa_hIL6 --gpu 1
#For transfer learning, fill the antigen and antibody sequences in the two datasets with the same length
python utools/feature_encoder.py --dataset CoVAbDab --gpu 1

python utools/protein_feature.py --dataset HIV --gpu 1
python utools/protein_feature.py --dataset SAbDab --gpu 1 --use_cdr
python utools/protein_feature.py --dataset AVIDa_hIL6 --gpu 1
#For transfer learning, fill the antigen and antibody sequences in the two datasets with the same length
python utools/protein_feature.py --dataset CoVAbDab --gpu 1
